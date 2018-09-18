import dlmark as dm
import mxnet as mx
from mxnet import nd
import time, os
import numpy as np
import json
import gluoncv as gcv
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform

models = gcv.model_zoo.model_store.pretrained_model_list()
yolo_models = [x for x in models if x.startswith('yolo') and x.endswith('coco')]

def get_map(model_name, data_shape):
    batch_size = 20  # divides 5000
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    model_name += '@' + str(data_shape)
    dataset = dm.image.COCOVal2017(batch_size, YOLO3DefaultValTransform(data_shape, data_shape),
        'yolo_default_%d'%(data_shape))
    val_dataset = gcv.data.COCODetection(splits='instances_val2017', skip_empty=False)
    metric = gcv.utils.metrics.coco_detection.COCODetectionMetric(
            val_dataset, '/tmp/{}_eval'.format(model_name), cleanup=True,
            data_shape=(data_shape, data_shape))
    ctx = mx.gpu(0)
    net.collect_params().reset_ctx(ctx)
    metric.reset()
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_shape=True, static_alloc=True)
    for ib in range(len(dataset)):
        batch = dataset[ib]
        x = batch[0].as_in_context(ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        for x in [x]:
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, data_shape))
        metric.update(det_bboxes, det_ids, det_scores)

    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':batch_size,
        'map':'{:.2f}'.format(float(metric.get()[1][-1])),
        'workload':'Inference',
    }

def benchmark_map():
    device_name = dm.utils.nv_gpu_name(0).replace(' ', '-').lower()
    results = []
    for model_name in yolo_models:
        for dshape in [320, 416, 608]:
            print(model_name, dshape)
            res, _ = dm.benchmark.run_with_separate_process(
                get_map, model_name, dshape
            )
            results.append(res)
            with open(os.path.join(os.path.dirname(__file__), 'yolo_'+device_name+'_map.json'), 'w') as f:
                json.dump(results, f)

benchmark_map()

def get_throughput(model_name, batch_size, data_shape):
    ctx = mx.gpu(0)
    device_name = dm.utils.nv_gpu_name(0)
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_shape=True, static_alloc=True)
    mem = dm.utils.nv_gpu_mem_usage()

    # warm up, need real data
    dataset = dm.image.COCOVal2017(batch_size, YOLO3DefaultValTransform(data_shape, data_shape),
        'yolo_default_%d'%(data_shape))
    X = dataset[0][0].as_in_context(ctx)
    # X = np.random.uniform(low=-254, high=254, size=(batch_size,3, data_shape,data_shape))
    # X = _preprocess(X).as_in_context(ctx)
    Y = net(X)
    nd.waitall()
    YY = Y[0][0].asnumpy()

    # iterate mutliple times
    iters = 1000 // batch_size
    tic = time.time()
    for _ in range(iters):
        Y = net(X)
        nd.waitall()
        YY = Y[0][0].asnumpy()

    throughput = iters*batch_size/(time.time()-tic)

    return {
        'device':device_name,
        'model':model_name + '@' + str(data_shape),
        'batch_size':batch_size,
        'throughput':throughput,
        'workload':'Inference',
        'device_mem':dm.utils.nv_gpu_mem_usage() - mem
    }

def benchmark_throughput():
    save = dm.benchmark.SaveResults(postfix=dm.utils.nv_gpu_name(0))
    for model_name in yolo_models:
        print(model_name)
        for dshape in [320, 416, 608]:
            batch_sizes = [1,2,4,8,16,20,32,64,128,256]
            for batch_size in batch_sizes:
                res, exitcode = dm.benchmark.run_with_separate_process(
                    get_throughput, model_name, batch_size, dshape
                )
                if exitcode:
                    break
                save.add(res)

benchmark_throughput()

def _try_batch_size(net, batch_size, data_shape, ctx, X):
    print('Try batch size', batch_size)
    def _run():
        net.collect_params().reset_ctx(ctx)
        XX = X.tile(reps=(batch_size, 1, 1, 1)).as_in_context(ctx)
        y = net(XX)
        nd.waitall()
        yy = y[0].asnumpy()

    _, exitcode = dm.benchmark.run_with_separate_process(_run)
    return exitcode == 0

def find_largest_batch_size(net, data_shape, X):
    upper = 1024
    lower = 1
    ctx = mx.gpu(0)
    while True:
        if _try_batch_size(net, upper, data_shape, ctx, X):
            upper *= 2
        else:
            break

    while (upper - lower) > 1:
        cur = (upper + lower) // 2
        if _try_batch_size(net, cur, data_shape, ctx, X):
            lower = cur
        else:
            upper = cur

    return lower

def benchmark_max_batch_size():
    save = dm.benchmark.SaveResults()
    device_name = dm.utils.nv_gpu_name(0)
    for model_name in yolo_models:
        print(model_name)
        net = gcv.model_zoo.get_model(model_name, pretrained=True)
        for data_shape in [320, 416, 608]:
            dataset = dm.image.COCOVal2017(1, YOLO3DefaultValTransform(data_shape, data_shape),
                'yolo_default_%d'%(data_shape))
            X = dataset[0][0]
            save.add({
                'device':device_name,
                'model':model_name + '@' + str(data_shape),
                'batch_size':find_largest_batch_size(net, (3,data_shape,data_shape), X),
                'workload':'Inference',
            })

benchmark_max_batch_size()
