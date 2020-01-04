import dlmark as dm
import mxnet as mx
from mxnet import nd
import time, os
import numpy as np
import json
import gluoncv as gcv
from mxnet import gluon
from gluoncv.data import imagenet
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import Block, HybridBlock

import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from gluoncv.data.transforms.pose import transform_preds, get_final_preds, flip_heatmap, heatmap_to_coord_alpha_pose
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform, SimplePoseDefaultValTransform
from gluoncv.utils.metrics.coco_keypoints import COCOKeyPointsMetric
from gluoncv.data.transforms.presets.alpha_pose import AlphaPoseDefaultValTransform

def _preprocess(X):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    X = nd.array(X).transpose((0,3,1,2))
    return (X.astype('float32') / 255 - rgb_mean) / rgb_std

whitelist = ['simple', 'alpha', 'mobile']
model_list = [x for x in gcv.model_zoo.pretrained_model_list() if x.split('_')[0].lower() in whitelist]
model_list = sorted(model_list)

def get_accuracy(model_name):
    batch_size = 64
    # dataset = dm.image.ILSVRC12Val(batch_size, 'http://xx/', root='/home/ubuntu/imagenet_val/')
    num_workers = dm.utils.get_cpu_count()
    num_joints = 17

    num_gpus = 4
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    ctx = context

    def get_data_loader(data_dir, batch_size, num_workers, input_size):

        def val_batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                              batch_axis=0, even_split=False)
            return tuple([data] + batch[1:])

        val_dataset = mscoco.keypoints.COCOKeyPoints(data_dir, splits=('person_keypoints_val2017'))

        meanvec = [float(i) for i in '0.485,0.456,0.406'.split(',')]
        stdvec = [float(i) for i in '0.229,0.224,0.225'.split(',')]
        if model_name.startswith('simple') or model_name.startswith('mobile'):
            transform_val = SimplePoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                          joint_pairs=val_dataset.joint_pairs,
                                                          image_size=input_size,
                                                          mean=meanvec,
                                                          std=stdvec)
        else:
            transform_val = AlphaPoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                         joint_pairs=val_dataset.joint_pairs,
                                                         image_size=input_size)
        val_data = gluon.data.DataLoader(
            val_dataset.transform(transform_val),
            batch_size=batch_size, shuffle=False, last_batch='keep',
            num_workers=num_workers)

        return val_dataset, val_data, val_batch_fn

    input_size = [int(i) for i in '320,256'.split(',')] if 'alpha_pose' in model_name else [int(i) for i in '256,192'.split(',')]
    val_dataset, val_data, val_batch_fn = get_data_loader('~/.mxnet/datasets/coco', batch_size,
                                                          num_workers, input_size)
    val_metric = COCOKeyPointsMetric(val_dataset, 'coco_keypoints',
                                     data_shape=tuple(input_size),
                                     in_vis_thresh=0)

    use_pretrained = True
    net = get_model(model_name, ctx=context, pretrained=use_pretrained)
    net.hybridize(static_alloc=True, static_shape=True)

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    val_metric.reset()
    
    flip_test = False

    from tqdm import tqdm
    for batch in tqdm(val_data):
        try:
            data, scale, center, score, imgid = val_batch_fn(batch, ctx)
        except:
            data, scale_box, score, imgid = val_batch_fn(batch, ctx)

        outputs = [net(X) for X in data]
        if flip_test:
            data_flip = [nd.flip(X, axis=3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip)/2 for o, o_flip in zip(outputs, outputs_flipback)]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[o.as_in_context(mx.cpu()) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())
        if model_name.startswith('simple_pose') or model_name.startswith('mobile_pose'):
            preds, maxvals = get_final_preds(outputs_stack, center.asnumpy(), scale.asnumpy())
        else:
            preds, maxvals = heatmap_to_coord_alpha_pose(outputs_stack, scale_box)
        val_metric.update(preds, maxvals, score, imgid)

    res = val_metric.get()
    print(res)

    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':batch_size,
        'accuracy':res[1],
        'workload':'Inference',
    }

def benchmark_accuracy():
    device_name = dm.utils.nv_gpu_name(0).replace(' ', '-').lower()
    results = []
    for model_name in model_list:
        print(model_name)
        res, _ = dm.benchmark.run_with_separate_process(
            get_accuracy, model_name
        )
        results.append(res)
        with open(os.path.join(os.path.dirname(__file__), 'pose_'+device_name+'_accuracy.json'), 'w') as f:
            json.dump(results, f)

def get_throughput(model_name, batch_size):
    ctx = mx.gpu(0)
    device_name = dm.utils.nv_gpu_name(0)
    # net = modelzoo[model_name](pretrained=True)
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True, static_shape=True)
    mem = dm.utils.nv_gpu_mem_usage()

    # warm up
    if model_name.startswith('simple_pose') or model_name.startswith('mobile_pose'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,256,192,3))
    else:
        X = np.random.uniform(low=-254, high=254, size=(batch_size,320,256,3))
    X = _preprocess(X).as_in_context(ctx)
    net(X).wait_to_read()

    # iterate mutliple times
    iters = 1000 // batch_size
    tic = time.time()
    device_mem = 0
    device_mem_count = 0
    ttime = 0.
    tic = time.time()
    for _ in range(iters):
        YY = net(X)
        YY.wait_to_read()
    nd.waitall()
    ttime = time.time() - tic
    throughput = iters*batch_size/ttime

    return {
        'device':device_name,
        'model':model_name,
        'batch_size':batch_size,
        'throughput':throughput,
        'workload':'Inference',
        'device_mem': dm.utils.nv_gpu_mem_usage() - mem
    }

def benchmark_throughput():
    save = dm.benchmark.SaveResults(postfix=dm.utils.nv_gpu_name(0))
    for model_name in model_list:
        print(model_name)
        # batch_sizes = [1,2,4,8,16,32,64,128,256]
        batch_sizes = [64]
        for batch_size in batch_sizes:
            res, exitcode = dm.benchmark.run_with_separate_process(
                get_throughput, model_name, batch_size
            )
            if exitcode:
                break
            save.add(res)

def _try_batch_size(net, batch_size, data_shape, ctx):
    print('Try batch size', batch_size)
    def _run():
        net.collect_params().reset_ctx(ctx)
        X = nd.random.uniform(shape=(batch_size, *data_shape), ctx=ctx)
        y = net(X)
        nd.waitall()

    _, exitcode = dm.benchmark.run_with_separate_process(_run)
    return exitcode == 0

def find_largest_batch_size(net, data_shape):
    upper = 1024
    lower = 1
    ctx = mx.gpu(0)
    while True:
        if _try_batch_size(net, upper, data_shape, ctx):
            upper *= 2
        else:
            break

    while (upper - lower) > 1:
        cur = (upper + lower) // 2
        if _try_batch_size(net, cur, data_shape, ctx):
            lower = cur
        else:
            upper = cur

    return lower

def benchmark_max_batch_size():
    save = dm.benchmark.SaveResults()
    device_name = dm.utils.nv_gpu_name(0)
    for model_name in model_list:
        print(model_name)
        # net = modelzoo[model_name](pretrained=True)
        net = gcv.model_zoo.get_model(model_name, pretrained=True)
        save.add({
            'device':device_name,
            'model':model_name,
            'batch_size':find_max_batch_size(net, (3,224,224)),
            'workload':'Inference',
        })

benchmark_accuracy()
benchmark_throughput()
# benchmark_max_batch_size()
