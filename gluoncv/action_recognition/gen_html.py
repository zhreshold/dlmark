import dlmark as dm

thr = dm.benchmark.load_results('ar.py__benchmark_throughput*json')

models = thr.model.unique()
devices = thr.device.unique()
print(models, devices)

from dlmark import plot
from bokeh.plotting import show, output_notebook
output_notebook()

data = thr[thr.device==devices[0]]
# show(plot.batch_size_vs_throughput_grid(data, models[::4]))

def load_results(fname):
    import pandas as pd
    import json
    import glob
    data = pd.DataFrame()
    for fn in glob.glob(fname):
        with open(fn, 'r') as f:
            result = json.load(f)
            result = [x for x in result if x is not None and not x['model'].endswith('v1s')]
            data = data.append(result)
    return data

def split_number(s):
    if s.startswith('inception'):
        return s.split('_')[:2]
    if s.startswith('i3d'):
        return s.split('_')[:2]
    if s.startswith('slowfast'):
        return s.split('_')[:2]
    import re
    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)
    if match:
        items = list(match.groups())
        if items[0] == 'mobilenetv':
            items[0] += str(items[1])
        if items[0] == 'resnet':
            if '0.' in s.split('_')[-1]:
                items[0] += '_pruned'
            else:
                items[0] += '_' + s.split('_')[-2]
        return items
    return [s]

paper_results = {
    'x': 1
}

import numpy as np
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap
from bokeh.models import LogTickFormatter, Span, Whisker, TeeHead
import pandas as pd


def throughput_vs_accuracy(data):
    assert 'accuracy' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
#     colors = palettes.Category20[max(len(models),3)]
    colors = palettes.Viridis256[max(len(models),3)]
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
    
    data['paper'] = data.index.map(paper_results)
#     data['paper_diff_percent'] = (data['accuracy'] - data['paper']) * 100
    data['paper_diff_percent'] = pd.Series(["{0:.2f}%".format((val1-val2) * 100) if val1 > val2 else 'N/A' for val1, val2 in zip(data['accuracy'], data['paper'])], index = data.index)

    p = figure(plot_width=600, plot_height=600, 
               x_axis_type="log", y_axis_type="log", active_drag="pan", toolbar_location='above')
    source = ColumnDataSource(data)

    p.scatter(x='throughput', y='accuracy', legend=model,
              size=size, color=index_cmap, source=source)
    
#     resnet50_ref = Span(location=0.7569,
#                               dimension='width', line_color=colors[models.index('resnet_v1')],
#                               line_dash='dashed', line_width=3)
#     p.add_layout(resnet50_ref)
    for ic, row in enumerate(data.iterrows()):
        idx = models.index(row[1]['model_prefix'])
        w_data = dict(base=[row[1].throughput], lower=[row[1].paper], upper=[row[1].accuracy])
        w_source = ColumnDataSource(data=w_data)
#         print(w_source.data)
        w = Whisker(source=w_source, base="base", upper="upper", lower="lower", dimension='height', line_color=colors[idx], lower_head=TeeHead(line_color=colors[idx]))
    #     print(w)
        p.add_layout(w)

    p.xaxis.axis_label = '#samples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Accuracy'

    p.legend.label_text_font_size = '12pt'
    p.legend[0].location = "bottom_left"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"

    toolstips = [("Model", "@model"),
                 ("Throughput", "@throughput"),
                 ("Accuracy", "@accuracy"),
                 ("Improvement over Reference", "@paper_diff_percent")]
    if 'device_mem' in data.columns:
        toolstips.append(["Device memory", "@device_mem MB"])
    p.add_tools(HoverTool(tooltips=toolstips))
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "hide"
    # p.xaxis[0].formatter = LogTickFormatter()

    return p

def make_dataset(data, model_list):
    data = data.copy()
    data = data[data.index.isin(model_list)]
    assert 'accuracy' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
#     colors = palettes.Category20[max(len(models),3)]
#     colors = np.random.choice(palettes.Viridis256, size=max(len(models),3))
    colors = palettes.Category20[20]
    index_cmap = factor_cmap(model, palette=colors, factors=models, end=1)
    data['color'] = [colors[models.index(c)] for c in data['model_prefix']]

    if ('device_mem' in data.columns and
        not 'size' in data.columns):
        size = np.sqrt(data.device_mem.values)
        data['size'] = 30 * size / size.max()

    if 'size' in data.columns:
        pass
    else:
        data['size'] = [10 for m in data.index]
    
    data['paper'] = data.index.map(paper_results.get)
#     data['paper_diff_percent'] = (data['accuracy'] - data['paper']) * 100
    data['paper_diff_percent'] = pd.Series(["{0:.2f}%".format((val1-val2) * 100) if val2 and val1 > val2 else 'N/A' for val1, val2 in zip(data['accuracy'], data['paper'])], index = data.index)
    data = data.fillna(0)
    source = ColumnDataSource(data)
    return source

def make_plot(src):
    from bokeh.models import CustomJS
    data = src.data
    p = figure(plot_width=600, plot_height=600,
               x_axis_type="log", active_drag="pan", toolbar_location='above', tools="pan,wheel_zoom,box_zoom,save,reset",)

#     p.circle(x='throughput', y='accuracy', legend='model_prefix',
#               size="size", color='color', source=src)

    toolstips = [("Model", "@model"),
                 ("Throughput", "@throughput"),
                 ("Accuracy", "@accuracy"),
                 ("Improvement over Reference", "@paper_diff_percent"),
                 ("Device memory", "@device_mem MB")]
    
    uniq_model = list(set(data['model_prefix']))
    uniq_model.sort()
    
    legend_map = dict()
    
    for current_prefix in uniq_model:
        indices = [i for i, x in enumerate(data['model_prefix']) if x == current_prefix]
        cc = {k: [v[i] for i in indices] for k, v in data.items()}
        current_source = ColumnDataSource(pd.DataFrame(cc))
        pr = p.circle(x='throughput', y='accuracy', legend='model_prefix', size="size", color='color', source=current_source)
        legend_map[current_prefix] = pr
        
    whisker_map = dict()
#     print(len(p.legend[0].items))
    for ic, row in enumerate(data['model_prefix']):
#         idx = legend_map.index(row)
        if not data['paper'][ic]:
            continue
        w_data = dict(base=[data['throughput'][ic]], lower=[data['paper'][ic]], upper=[data['accuracy'][ic]])
        w_source = ColumnDataSource(data=w_data)
#         print(float(data['paper'][ic]), w_source.data, data['color'][ic])
        w = Whisker(source=w_source, base="base", upper="upper", lower="lower", dimension='height', line_color=data['color'][ic], lower_head=TeeHead(size=10, line_color=data['color'][ic]), upper_head=TeeHead(size=1, line_color=data['color'][ic]))
#         print(type(w))
#         print(type(p.legend[0].items[idx].renderers))
#         p.legend[0].items[idx].renderers.append([w])
        whisker_map[row] = [w] if row not in whisker_map else whisker_map[row] + [w]
#         
        p.add_layout(w)
        
    for name in whisker_map.keys():
        assert name in legend_map.keys()
        c = legend_map[name]
        w = whisker_map[name]
        c.js_on_change('visible', CustomJS(args=dict(c=c, w=w), code="""
        // get data source from Callback args
        console.log('enter', c.visible, w.visible);
        w.forEach(function(element) {
          element.visible = c.visible;
        });
        // w.visible = c.visible; 
        """))
        # """

    p.xaxis.axis_label = '#samples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Accuracy'

    p.legend.label_text_font_size = '1em'
    p.legend[0].location = "bottom_left"
    p.xaxis.axis_label_text_font_size = "1.5em"
    p.yaxis.axis_label_text_font_size = "1.5em"
    p.title.text_font_size = '100%'
    p.xaxis.axis_label_text_font_size = "1em"
    p.xaxis.major_label_text_font_size = "0.5em"
    p.yaxis.axis_label_text_font_size = "1em"
    p.yaxis.major_label_text_font_size = "0.5em"

    
    p.add_tools(HoverTool(tooltips=toolstips))
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "hide"
    p.toolbar.logo = None
#     for ppp in p.legend[0].items:
#         print(ppp.renderers)
    # p.xaxis[0].formatter = LogTickFormatter()
    return p

def update(attr, old, new):
    print('update!!!')
    models_to_plot = [model_selection.labels[i] for i in 
                        model_selection.active]
    new_src = make_dataset(data, models_to_plot)
    src.data.update(ColumnDataSource(new_src).data)
#     pp = make_plot(new_src, pp)

acc = load_results('ar_tesla-v100-sxm2-16gb_accuracy.json')
print(acc)

data = thr[(thr.model.isin(acc.model)) & 
           (thr.batch_size.isin(acc.batch_size)) &
           (thr.device.isin(acc.device))]
data = data.set_index('model').join(acc[['model','accuracy']].set_index('model'))
# data['model_prefix'] = [i[:i.rfind('-')] if i.rfind('-') > 0 else i for i in data.index]
data['model_prefix'] = [split_number(i)[0] for i in data.index]

# from bokeh.models.widgets import CheckboxGroup
# # Create the checkbox selection element, available carriers is a  
# # list of all airlines in the data
# active_ = list(range(len(data.index.values)))
# model_selection = CheckboxGroup(labels=data.index.values.tolist(), 
#                                 active = [0, 1])

# pp = throughput_vs_accuracy(data)
src = make_dataset(data, data.index.values.tolist())
pp = make_plot(src)
pp.sizing_mode = 'scale_width'
# from bokeh.layouts import widgetbox
# show(widgetbox(model_selection))

from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

html = file_html(pp, CDN, "action_recognition")
html = html.replace('Bokeh.set_log_level("info");', '''Bokeh.set_log_level("info"); window.onload = function () { window.dispatchEvent(new Event('resize')); };''')

import os
with open(os.path.join(os.path.abspath(''), 'ar_throughputs.html'), 'wt') as f:
    f.write(html)
