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

import argparse, time, logging, os, sys, math
import gc

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import gluoncv as gcv
from mxnet import gluon, nd, gpu, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter

from gluoncv.data.transforms import video
from gluoncv.data import ucf101, kinetics400
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load
from gluoncv.data.dataloader import tsn_mp_batchify_fn

def _preprocess(X):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    X = nd.array(X).transpose((0,3,1,2))
    return (X.astype('float32') / 255 - rgb_mean) / rgb_std

whitelist = ['kinetics400']
model_list = [x for x in gcv.model_zoo.pretrained_model_list() if ('kinetics400' in x)]

def get_accuracy(model_name):

    table = {
    'inceptionv1_kinetics400': 0.691,
    'inceptionv3_kinetics400': 0.725,
    'resnet18_v1b_kinetics400': 0.655,
    'resnet34_v1b_kinetics400': 0.691,
    'resnet50_v1b_kinetics400': 0.6990000000000001,
    'resnet101_v1b_kinetics400': 0.713,
    'resnet152_v1b_kinetics400': 0.715,
    'i3d_inceptionv1_kinetics400': 0.718,
    'i3d_inceptionv3_kinetics400': 0.736,
    'i3d_resnet50_v1_kinetics400': 0.74,
    'i3d_resnet101_v1_kinetics400': 0.7509999999999999,
    'i3d_nl5_resnet50_v1_kinetics400': 0.752,
    'i3d_nl10_resnet50_v1_kinetics400': 0.753,
    'i3d_nl5_resnet101_v1_kinetics400': 0.76,
    'i3d_nl10_resnet101_v1_kinetics400': 0.7609999999999999,
    'slowfast_4x16_resnet50_kinetics400': 0.753,
    'slowfast_8x8_resnet50_kinetics400': 0.7659999999999999,
    'slowfast_8x8_resnet101_kinetics400': 0.772,
    'resnet50_v1b_ucf101': 0.8370000000000001,
    'i3d_resnet50_v1_ucf101': 0.8390000000000001,
    'i3d_resnet50_v1_ucf101': 0.9540000000000001,
    'resnet50_v1b_hmdb51': 0.552,
    'i3d_resnet50_v1_hmdb51': 0.485,
    'i3d_resnet50_v1_hmdb51': 0.7090000000000001,
    'resnet50_v1b_sthsthv2': 0.355,
    'i3d_resnet50_v1_sthsthv2': 0.506,
    }    

    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':2,
        'accuracy':table.get(model_name, 0),
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
        with open(os.path.join(os.path.dirname(__file__), 'ar_'+device_name+'_accuracy.json'), 'w') as f:
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
        X = np.random.uniform(low=-254, high=254, size=(batch_size,3,299,299))
    elif model_name.startswith('resnet'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,3,224,224))
    elif model_name.startswith('i3d'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,3, 32, 224, 224)) 
    elif model_name.startswith('slowfast_4x16'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,3,36,224,224))
    elif model_name.startswith('slowfast_8x8'):
        X = np.random.uniform(low=-254, high=254, size=(batch_size,3,40,224,224))
    else:
        raise ValueError('Unknown model:' + model_name)
    X = mx.nd.array(X).as_in_context(ctx)
    net(X).wait_to_read()

    # iterate mutliple times
    iters = 100 // batch_size
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
        batch_sizes = [2]
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
