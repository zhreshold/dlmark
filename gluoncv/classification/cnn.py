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

def _preprocess(X):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    X = nd.array(X).transpose((0,3,1,2))
    return (X.astype('float32') / 255 - rgb_mean) / rgb_std

blacklist = ['faster', 'ssd', 'yolo3', 'fcn', 'psp', 'mask', 'cifar', 'deeplab', 'simple_pose', 'alpha_pose']
model_list = [x for x in gcv.model_zoo.pretrained_model_list() if x.split('_')[0].lower() not in blacklist]
action_list = ['kinetics400', 'ucf101']
model_list = [x for x in model_list if x.split('_')[1].lower() not in action_list]

def get_accuracy(model_name):
    batch_size = 64
    # dataset = dm.image.ILSVRC12Val(batch_size, 'http://xx/', root='/home/ubuntu/imagenet_val/')
    num_workers = dm.utils.get_cpu_count()
    ctx = mx.gpu(0)
    # net = modelzoo[model_name](pretrained=True)
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx, classes=1000)
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True, static_shape=True)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if model_name.startswith('inception'):
        transform_test = transforms.Compose([
        transforms.Resize(342, keep_ratio=True),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
        ])
    else:
        transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])

    dataset = gluon.data.DataLoader(
        imagenet.classification.ImageNet(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n, acc = 0, 0
    for batch in dataset:
        X = batch[0].as_in_context(ctx)
        y = batch[1].as_in_context(ctx)
        yhat = net(X)
        acc += nd.sum(yhat.argmax(axis=1).astype('int64')==y).asscalar()
        n += X.shape[0]
        if n > 5e4:
            break

    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':batch_size,
        'accuracy':acc/n,
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
        with open(os.path.join(os.path.dirname(__file__), 'cnn_'+device_name+'_accuracy.json'), 'w') as f:
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
    if model_name.startswith('inception'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,299,299,3))
    else:
        X = np.random.uniform(low=-254, high=254, size=(batch_size,224,224,3))
    X = _preprocess(X).as_in_context(ctx)
    net(X).wait_to_read()

    # iterate mutliple times
    iters = 1000 // batch_size
    tic = time.time()
    device_mem = 0
    device_mem_count = 0
    ttime = 0.
    for _ in range(iters):
        tic = time.time()
        YY = net(X)
        YY.wait_to_read()
        ttime += time.time() - tic
        device_mem += dm.utils.nv_gpu_mem_usage() - mem
        device_mem_count += 1
    nd.waitall()
    throughput = iters*batch_size/ttime

    return {
        'device':device_name,
        'model':model_name,
        'batch_size':batch_size,
        'throughput':throughput,
        'workload':'Inference',
        'device_mem': device_mem / float(device_mem_count)
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
