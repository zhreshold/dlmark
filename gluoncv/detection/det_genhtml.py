import dlmark as dm
import pandas as pd
import numpy as np
import os

prefixs = ['ssd', 'yolo', 'faster_rcnn', 'center_net']
thrs = [dm.benchmark.load_results(os.path.join(os.path.dirname(__file__), prefix) + '.py__benchmark_throughput*json') for prefix in prefixs]
thr = pd.concat(thrs)

models = thr.model.unique()
devices = thr.device.unique()
print(models, devices)

from dlmark import plot
from bokeh.plotting import show, output_notebook
output_notebook()

data = thr[(thr.device==devices[0]) & (thr.batch_size.isin([1,2,4,8,16,32,64,128]))]
bs = pd.concat([dm.benchmark.load_results(prefix + '.py__benchmark_max_batch_size.json') for prefix in prefixs])


paper_results = {
    'faster_rcnn_resnet50_v1b_coco': 36.5,
    'yolo3_darknet53_coco@608': 33.0,
    'yolo3_darknet53_coco@416': 31.0,
    'yolo3_darknet53_coco@320': 28.6,
}

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
        
    data['paper'] = data.index.map(paper_results)
#     data['paper_diff_percent'] = (data['accuracy'] - data['paper']) * 100
    data['paper_diff_percent'] = pd.Series(["{0:.2f}%".format((val1-val2) * 100) if val1 > val2 else 'N/A' for val1, val2 in zip(data['accuracy'], data['paper'])], index = data.index)
    data = data.fillna(0)

    p = figure(plot_width=600, plot_height=600, title='Inference Throughput vs. mAP on COCO',
               x_axis_type="log", active_drag="pan", toolbar_location='above')

    source = ColumnDataSource(data)

    p.circle(x='throughput', y='map', legend=model,
              size=size, color=index_cmap, source=source)
    
    for ic, row in enumerate(data['model_prefix']):
        if not data['paper'][ic]:
            continue
        w_data = dict(base=[data['throughput'][ic]], lower=[data['paper'][ic]], upper=[data['accuracy'][ic]])
        w_source = ColumnDataSource(data=w_data)
        w = Whisker(source=w_source, base="base", upper="upper", lower="lower", dimension='height', line_color=data['color'][ic], lower_head=TeeHead(size=10, line_color=data['color'][ic]), upper_head=TeeHead(size=1, line_color=data['color'][ic]))
        whisker_map[row] = [w] if row not in whisker_map else whisker_map[row] + [w]
        p.add_layout(w)
    
    p.legend.label_text_font_size = '1.5vw'
    p.legend[0].location = "bottom_left"

    p.xaxis.axis_label = '#samples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'mAP'
    
    p.xaxis.axis_label_text_font_size = "2vw"
    p.yaxis.axis_label_text_font_size = "2vw"
    p.title.text_font_size = "2vw"

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
    p.sizing_mode = 'scale_width'
    return p

import numpy as np
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap
from bokeh.models import LogTickFormatter, Span, Whisker, TeeHead
import pandas as pd

def make_dataset(data, model_list):
    from bokeh import palettes
    from bokeh.transform import factor_cmap
    data = data.copy()
    data.map = data.map.astype(float)
    data = data[data.index.isin(model_list)]
    assert 'map' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
    colors = palettes.Category10[max(len(models),3)]
    index_cmap = factor_cmap(model, palette=colors, factors=models, end=1)
    data['color'] = [colors[models.index(c)] for c in data['model_prefix']]

    if ('device_mem' in data.columns and 'batch_size' in data.columns and
        'device_mem_per_batch' not in data.columns):
        data['device_mem_per_batch'] = data['device_mem'] / data['batch_size']
    if ('device_mem_per_batch' in data.columns and
        not 'size' in data.columns):
        size = np.sqrt(data.device_mem_per_batch.values)
        data['size'] = 30 * size / size.max()

    if 'size' in data.columns:
        pass
    else:
        data['size'] = [10 for m in data.index]
        
    data['paper'] = data.index.map(paper_results)
#     data['paper_diff_percent'] = (data['accuracy'] - data['paper']) * 100
    data['paper_diff_percent'] = pd.Series(["{0:.2f}%".format((float(val1)-val2)) if float(val1) > float(val2) else 'N/A' for val1, val2 in zip(data['map'], data['paper'])], index = data.index)
    data = data.fillna(0)
    source = ColumnDataSource(data)
    return source

def make_plot(src):
    
    from bokeh.models import CustomJS
    data = src.data
    p = figure(plot_width=600, plot_height=600,
               x_axis_type="log", active_drag="pan", toolbar_location='above',tools="pan,wheel_zoom,box_zoom,save,reset",)

    toolstips = [("Model", "@model"),
                 ("Throughput", "@throughput"),
                 ("mAP", "@map"),
                 ("Improvement over Reference", "@paper_diff_percent"),
                 ("Device memory Per Batch", "@device_mem_per_batch MB"),
                 ("Best throughput @Batch size", "@best_throughput_batch_size")]
    
    uniq_model = list(set(data['model_prefix']))
    uniq_model.sort()
    
    legend_map = dict()
    
    for current_prefix in uniq_model:
        indices = [i for i, x in enumerate(data['model_prefix']) if x == current_prefix]
        cc = {k: [v[i] for i in indices] for k, v in data.items()}
        current_source = ColumnDataSource(pd.DataFrame(cc))
        pr = p.circle(x='throughput', y='map', legend='model_prefix', size="size", color='color', source=current_source)
        legend_map[current_prefix] = pr
        
    whisker_map = dict()
#     print(len(p.legend[0].items))
    for ic, row in enumerate(data['model_prefix']):
#         idx = legend_map.index(row)
        if not data['paper'][ic]:
            continue
        w_data = dict(base=[data['throughput'][ic]], lower=[data['paper'][ic]], upper=[data['map'][ic]])
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
        // console.log('enter', c.visible, w.visible);
        w.forEach(function(element) {
          element.visible = c.visible;
        });
        // w.visible = c.visible;
        """))

    p.xaxis.axis_label = '#samples/sec'
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'mAP'

    p.legend[0].location = "bottom_left"
    p.legend.label_text_font_size = '1em'
    p.xaxis.axis_label_text_font_size = "1.5em"
    p.title.text_font_size = '1em'
    p.yaxis.axis_label_text_font_size = "1em"
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

maps = pd.concat([dm.benchmark.load_results(os.path.join(os.path.dirname(__file__), prefix) + '*map.json') for prefix in prefixs])
thr = thr.reset_index(drop=True)
thr_sorted = thr.sort_values(by='throughput', ascending=False).drop_duplicates(['model'])
thr_sorted['best_throughput_batch_size'] = thr.batch_size[thr_sorted.index]

data = thr_sorted[(thr_sorted.model.isin(maps.model)) &
           (thr_sorted.device.isin(maps.device))]
data1 = data.set_index('model').join(maps[['model','map', 'map_per_class']].set_index('model'))
data = data.set_index('model').join(maps[['model','map']].set_index('model'))
data['model_prefix'] = [i[:i.find('_')] if i.find('_') > 0 and i[:i.find('_')] in ('ssd', 'yolo3') else i[:i.find('_', i.find('_') + 1)] for i in data.index]
# print(data['model_prefix'])
# print(data['model_prefix']['faster_rcnn_resnet50_v1b_coco'])
# print(data['model_prefix'])

# pp = throughput_vs_map(data)
src = make_dataset(data, data.index.values.tolist())
pp = make_plot(src)
pp.sizing_mode = 'scale_width'

def plot_bar(data):
    import copy
    import numpy as np
    from bokeh import palettes
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, FactorRange
    from bokeh.layouts import gridplot
    from bokeh.transform import factor_cmap
    from bokeh.models import LogTickFormatter
    from bokeh.models import Legend
    from bokeh.palettes import Spectral11, Plasma256, Category20
    assert 'map_per_class' in data.columns
#     new_data = pd.concat([data.drop(['map_per_class'], axis=1), data['map_per_class'].apply(pd.Series)], axis=1)
#     print(new_data)
    keys = list(data.map_per_class[0].keys())
    keys.sort()
    keys = keys[::-1]
    p = figure(plot_width=500, plot_height=1000, y_range=keys,
               active_drag="pan", toolbar_location='above', tools="pan,wheel_zoom,box_zoom,save,reset",)
    
    colors = palettes.Category20b[20] + palettes.Category20c[20]

    for ir, row in enumerate(data.iterrows()):
        v = list([row[1]['map_per_class'][k] for k in keys])
        source = ColumnDataSource(data=dict(x=keys, value=v, model=[row[1].name for _ in keys]))
        p.circle(x='value', source=source, y='x', color=colors[ir], legend=data.index[ir], size=10)
    p.xgrid.grid_line_color = None
    p.legend.click_policy = "hide"
    
#     p.y_range.start = 0
#     p.yaxis.major_label_orientation = "vertical"
    

    new_legend = p.legend[0]
#     print((new_legend.items))
    N = 3
    num_legends = int(np.ceil(len(new_legend.items) / N))
    new_legends = []
    for i, item in enumerate(range(num_legends)):
        begin = i * N
        end = min(i * N + N, len(new_legend.items))
#         print(begin, end)
        ll = Legend(items=new_legend.items[begin:end],
                 location=(0, 0 * (i + 1)), orientation="horizontal")
        new_legends.append(ll)
    p.legend[0].items = []
#     p.legend[0].visible = None
#     p.add_layout(new_legend, 'below')
    for nl in new_legends:
#         print(type(nl))
        p.add_layout(nl, 'above')
    p.xaxis.axis_label = 'mAP per category(%)'
    p.sizing_mode = 'scale_width'
    p.legend.click_policy = "hide"
    toolstips = [("Model", "@model"), ("Category", "@x"), ("mAP", "@value")]
    p.add_tools(HoverTool(tooltips=toolstips))
    p.toolbar.logo = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = '0.5em'
    p.xaxis.axis_label_text_font_size = "1.5em"
    p.title.text_font_size = '1em'
    p.yaxis.axis_label_text_font_size = "1em"
    p.xaxis.axis_label_text_font_size = "1em"
    p.xaxis.major_label_text_font_size = "0.5em"
    p.yaxis.axis_label_text_font_size = "1em"
    p.yaxis.major_label_text_font_size = "0.8em"
    return p
    
    
p2 = plot_bar(data1)
p2.sizing_mode = 'scale_width'

from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

html = file_html(pp, CDN, "detection throughputs")
html = html.replace('Bokeh.set_log_level("info");', '''Bokeh.set_log_level("info"); window.onload = function () { window.dispatchEvent(new Event('resize')); };''')
# print(html)
with open(os.path.join(os.path.dirname(__file__), 'detection_throughputs.html'), 'wt') as f:
    f.write(html)
    
html2 = file_html(p2, CDN, "detection coco per class")
html2 = html2.replace('Bokeh.set_log_level("info");', '''Bokeh.set_log_level("info"); window.onload = function () { window.dispatchEvent(new Event('resize')); };''')
# print(html2)
with open(os.path.join(os.path.dirname(__file__), 'detection_coco_per_class.html'), 'wt') as f:
    f.write(html2)