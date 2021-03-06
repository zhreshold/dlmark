import numpy as np
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap
from bokeh.models import LogTickFormatter


def batch_size_vs_throughput(data, color='steelblue'):
    assert 'batch_size' in data.columns, data.columns
    assert 'throughput' in data.columns, data.columns
    p = figure(plot_width=250, plot_height=250,
               x_axis_type="log",
               toolbar_location=None, tools="")
    if 'size' in data.columns:
        size = 'size'
    else:
        size = 10
    source = ColumnDataSource(data=data)
    p.scatter(x='batch_size', y='throughput', size=size,
              color=color, source=source)
    p.xaxis.axis_label = 'batch size'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = '#examples / sec'
    p.xaxis[0].ticker = data.batch_size.values


    toolstips = [("Throughput", "@throughput"),
                 ("Batch size", "@batch_size")]
    if 'device_mem' in data.columns:
        toolstips.append(["Device memory", "@device_mem MB"])
    p.add_tools(HoverTool(tooltips=toolstips))
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    return p


def batch_size_vs_throughput_grid(data, models):
    assert 'batch_size' in data.columns, data.columns
    assert 'throughput' in data.columns, data.columns
    assert 'model' in data.columns, data.columns

    data = data.copy()
    if ('device_mem' in data.columns and
        not 'size' in data.columns):
        size = np.log(data.device_mem.values)
        data['size'] = 20 * size / size.max()
    # colors = palettes.Set2[max(len(models),3)]
    plots = []
    # for model, color in zip(models, colors):
    for model in models:
        x = data[data.model == model]
        p = batch_size_vs_throughput(x)
        if len(x) and 'device' in x.columns:
            p.title.text = model + " @ " + x.device.iloc[0]
        else:
            p.title.text = model
        plots.append(p)
    return gridplot(plots, ncols=2, toolbar_location="")

def throughput_vs_accuracy(data):
    assert 'accuracy' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
    colors = palettes.Category10[max(len(models),3)]
    index_cmap = factor_cmap(model, palette=colors, factors=models, end=1)

    data = data.copy()
    if ('device_mem' in data.columns and
        not 'size' in data.columns):
        size = np.sqrt(data.device_mem.values)
        data['size'] = 30 * size / size.max()

    if 'size' in data.columns:
        size = 'size'
    else:
        size = 10

    p = figure(plot_width=600, plot_height=500,
               toolbar_location=None, tools="", x_axis_type="log")
    source = ColumnDataSource(data)

    p.scatter(x='throughput', y='accuracy', legend=model,
              size=size, color=index_cmap, source=source)

    p.xaxis.axis_label = '#examples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Accuracy'

    toolstips = [("Model", "@model"),
                 ("Throughput", "@throughput"),
                 ("Accuracy", "@accuracy")]
    if 'device_mem' in data.columns:
        toolstips.append(["Device memory", "@device_mem MB"])
    p.add_tools(HoverTool(tooltips=toolstips))
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.legend.background_fill_alpha = 0
    # p.xaxis[0].formatter = LogTickFormatter()

    return p

def max_batch_size(data):
    models = data['model'].values
    batchs = data['batch_size'].values
    source = ColumnDataSource(data=data)
    title = "Maximal batch size"
    if 'device' in data.columns:
        dev = data.device.unique()
        assert len(dev) == 1, dev
        title += " @ " + dev[0]
    p = figure(x_range=models, plot_height=250, toolbar_location=None, title=title)
    p.vbar(x='model', top='batch_size', width=0.9, source=source,
        line_color='white')

    tooltips = [("Model", "@model"),
                 ("Max batch size", "@batch_size")]
    p.add_tools(HoverTool(tooltips=tooltips))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.xaxis.major_label_orientation = .5

    return p

def throughput_vs_device(data):
    assert 'device' in data.columns
    assert 'batch_size' in data.columns
    assert 'throughput' in data.columns

    data = data.sort_values(['device', 'batch_size'])
    x = [(dev, str(bs)) for dev, bs in zip(data.device, data.batch_size)]
    y = data.throughput
    title='Throughput vs devices'
    if 'model' in data:
        title += ' @ ' + data.model.iloc[0]
    source = ColumnDataSource(data=dict(x=x, y=y))
    p = figure(x_range=FactorRange(*x), plot_height=250, toolbar_location=None,
               title=title)
    p.vbar(x='x', top='y', width=.8, source=source)

    tooltips = [("Throughput", "@y")]
    p.add_tools(HoverTool(tooltips=tooltips))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.yaxis.axis_label = '#examples/sec'
    return p
