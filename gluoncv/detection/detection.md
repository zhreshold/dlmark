# GluonCV: detection

We benchmark the convolutional neural networks provided by the [GluonCV modelzoo](https://gluon-cv.mxnet.io/model_zoo/index.html#) for object detection.

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=49}
import dlmark as dm
import pandas as pd
import numpy as np

prefixs = ['ssd', 'yolo', 'faster_rcnn']

thrs = [dm.benchmark.load_results(prefix + '.py__benchmark_throughput*json') for prefix in prefixs]
thr = pd.concat(thrs)

models = thr.model.unique()
devices = thr.device.unique()
(models, devices)
```

Now we visualize the throughput for each network when increasing the batch sizes. We only use the results on the first device and show a quater of networks:

```{.python .input  n=2}
from dlmark import plot
from bokeh.plotting import show, output_notebook
output_notebook()

data = thr[(thr.device==devices[0]) & (thr.batch_size.isin([1,2,4,8,16,32,64,128]))]
show(plot.batch_size_vs_throughput_grid(data, models[::1]))
```

The throughput increases with the batch size in log scale. The device memory, as exepcted, also increases linearly with the batch size. But note that, due to the pooled memory mechanism in MXNet, the measured device memory usage might be different to the actual memory usdage.

One way to measure the actual device memory usage is finding the largest batch size we can run.

```{.python .input  n=3}
bs = pd.concat([dm.benchmark.load_results(prefix + '.py__benchmark_max_batch_size.json') for prefix in prefixs])
show(plot.max_batch_size(bs))
```

## Throughput on various hardware

```{.python .input  n=4}
#show(plot.throughput_vs_device(thr[(thr.model=='AlexNet')]))
```

```{.python .input  n=5}
#show(plot.throughput_vs_device(thr[(thr.model=='ResNet-v2-50')]))
```

### Prediction mAP versus throughput

We measture the prediction mean AP of each model using the MSCOCO 2017 validation dataset. Then plot the results together with the best throughput across various batch-sizes(if applicable). We colorize models from the same family with the same color.

```{.python .input  n=84}
def throughput_vs_map(data):
    import numpy as np
    from bokeh import palettes
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, FactorRange
    from bokeh.layouts import gridplot
    from bokeh.transform import factor_cmap
    from bokeh.models import LogTickFormatter
    assert 'map' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
    colors = palettes.Category10[max(len(models),3)]
    index_cmap = factor_cmap(model, palette=colors, factors=models, end=1)

    data = data.copy()
    if ('device_mem' in data.columns and 'batch_size' in data.columns and
        'device_mem_per_batch' not in data.columns):
        data['device_mem_per_batch'] = data['device_mem'] / data['batch_size']
    if ('device_mem_per_batch' in data.columns and
        not 'size' in data.columns):
        size = np.sqrt(data.device_mem_per_batch.values)
        data['size'] = 30 * size / size.max()

    data.map = data.map.astype(float)
    if 'size' in data.columns:
        size = 'size'
    else:
        size = 10

    p = figure(plot_width=600, plot_height=500,
               toolbar_location=None, tools="", x_axis_type="log")

    source = ColumnDataSource(data)

    p.circle(x='throughput', y='map', legend=model,
              size=size, color=index_cmap, source=source)

    p.xaxis.axis_label = '#examples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'mAP'

    toolstips = [("Model", "@model"),
                 ("Throughput", "@throughput"),
                 ("mAP", "@map")]
    if 'device_mem_per_batch' in data.columns:
        toolstips.append(["Device memory Per Batch", "@device_mem_per_batch MB"])
    if 'best_throughput_batch_size' in data.columns:
        toolstips.append(["Best throughput @Batch size", "@best_throughput_batch_size"])
    p.add_tools(HoverTool(tooltips=toolstips))
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.legend.background_fill_alpha = 0
    # p.xaxis[0].formatter = LogTickFormatter()

    return p
```

```{.python .input  n=85}
maps = pd.concat([dm.benchmark.load_results(prefix + '*map.json') for prefix in prefixs])
thr = thr.reset_index(drop=True)
thr_sorted = thr.sort_values(by='throughput', ascending=False).drop_duplicates(['model'])
thr_sorted['best_throughput_batch_size'] = thr.batch_size[thr_sorted.index]

data = thr_sorted[(thr_sorted.model.isin(maps.model)) &
           (thr_sorted.device.isin(maps.device))]
data = data.set_index('model').join(maps[['model','map']].set_index('model'))
data['model_prefix'] = [i[:i.find('_')] if i.find('_') > 0 else i for i in data.index]

show(throughput_vs_map(data))
```
