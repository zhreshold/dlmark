# Convolutional Neural Network

We benchmark the convolutional neural networks provided by the [Gluon modelzoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html).

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=1}
import dlmark as dm
from bokeh.plotting import show, output_notebook
import json
import pandas as pd
output_notebook()

!cat cnn*throughput.json >thr.json
with open('thr.json', 'r') as f:
    thr = pd.DataFrame(json.load(f))
    
models = thr.model.unique()
devices = thr.device.unique()
(models, devices)
```

Now we visualize the throughput for each network when increasing the batch sizes. We only use the results on the first device and show the first network:

```{.python .input  n=2}
data = thr[thr.device==devices[0]]
show(dm.plot.batch_size_vs_throughput_grid(data, models[:1]))
```

The throughput increases with the batch size in log scale. The device memory, as exepcted, also increases linearly with the batch size. But note that, due to the pooled memory mechanism in MXNet, the measured device memory usage 

similiar for the device memory usage.

```{.python .input  n=3}
show(dm.plot.batch_size_vs_throughput_grid(data, models[1:]))
```

### Prediction accuracy versus throughput

```{.python .input  n=4}
!cat cnn*accuracy.json >acc.json
with open('acc.json', 'r') as f:
    acc = pd.DataFrame(json.load(f))

data = thr[(thr.model.isin(acc.model)) &
           (thr.batch_size.isin(acc.batch_size))]
data = data.set_index('model').join(acc[['model','accuracy']].set_index('model'))
data['model_prefix'] = [i[:i.rfind('-')] if i.rfind('-') > 0 else i for i in data.index]

```

```{.python .input  n=5}
p = dm.plot.throughput_vs_accuracy(data)
show(p)
```