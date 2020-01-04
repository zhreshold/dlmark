import dlmark as dm
import mxnet as mx
from mxnet import nd
import time, os
import numpy as np
import json
import gluoncv
import gluoncv as gcv
from mxnet import gluon
from gluoncv.data import imagenet
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import Block, HybridBlock
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from tqdm import tqdm

def _preprocess(X):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    X = nd.array(X).transpose((0,3,1,2))
    return (X.astype('float32') / 255 - rgb_mean) / rgb_std

whitelist = ['fcn', 'psp', 'deeplab']
model_list = [x for x in gcv.model_zoo.pretrained_model_list() if x.split('_')[0].lower() in whitelist]
model_list = [x for x in model_list if x.endswith('coco')]

def get_accuracy(model_name):
    ctx = [mx.gpu(i) for i in range(4)]
    batch_size = 4
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # get dataset
    testset = get_segmentation_dataset(
        'coco', split='val', mode='testval', transform=input_transform)
    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    test_data = gluon.data.DataLoader(
        testset, batch_size, shuffle=False, last_batch='keep',
        batchify_fn=ms_batchify_fn, num_workers=32)
    model = get_model(model_name, pretrained=True)
    model.collect_params().reset_ctx(ctx=ctx)
#     model.hybridize(static_shape=True, static_alloc=True)
    evaluator = MultiEvalModel(model, testset.num_class, ctx_list=ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
        targets = [target.as_in_context(predicts[0].context) \
                   for target in dsts]
        metric.update(targets, predicts)
        pixAcc, mIoU = metric.get()
        tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':batch_size,
        'mIoU':mIoU,
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
        with open(os.path.join(os.path.dirname(__file__), 'semantic_segmentation_'+device_name+'_accuracy.json'), 'w') as f:
            json.dump(results, f)

def get_throughput(model_name, batch_size):
    ctx = mx.gpu(0)
    device_name = dm.utils.nv_gpu_name(0)
    # net = modelzoo[model_name](pretrained=True)
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True, static_shape=True)
    mem = dm.utils.nv_gpu_mem_usage()

    
    
    bs = batch_size
    num_iterations = 100
    input_shape = (bs, 3, 480, 480)
    size = num_iterations * bs
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=mx.gpu(0), dtype='float32')
    dry_run = 5

    with tqdm(total=size+dry_run*bs) as pbar:
        for n in range(dry_run + num_iterations):
            if n == dry_run:
                tic = time.time()
            outputs = net(data)
            for output in outputs:
                output.wait_to_read()
            pbar.update(bs)
    speed = size / (time.time() - tic)
    print('With batch size %d , %d batches, throughput is %f imgs/sec' % (bs, num_iterations, speed))

    return {
        'device':device_name,
        'model':model_name,
        'batch_size':batch_size,
        'throughput':speed,
        'workload':'Inference',
        'device_mem': dm.utils.nv_gpu_mem_usage() - mem
    }

def benchmark_throughput():
    save = dm.benchmark.SaveResults(postfix=dm.utils.nv_gpu_name(0))
    for model_name in model_list:
        print(model_name)
        # batch_sizes = [1,2,4,8,16,32,64,128,256]
        batch_sizes = [32]
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
