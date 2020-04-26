# Convolutional Neural Network

We benchmark the convolutional neural networks provided by the [Gluon modelzoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html).

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=1}
import dlmark as dm

thr = dm.benchmark.load_results('semantic_segmentation.py__benchmark_throughput*json')

models = thr.model.unique()
devices = thr.device.unique()
(models, devices)
```

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "(array(['psp_resnet101_coco', 'fcn_resnet101_coco',\n        'deeplab_resnet101_coco', 'deeplab_resnet152_coco'], dtype=object),\n array(['Tesla V100-SXM2-16GB'], dtype=object))"
  },
  "execution_count": 1,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now we visualize the throughput for each network when increasing the batch sizes. We only use the results on the first device and show a quater of networks:

```{.python .input  n=2}
from dlmark import plot
from bokeh.plotting import show, output_notebook
output_notebook()

data = thr[thr.device==devices[0]]
# show(plot.batch_size_vs_throughput_grid(data, models[::4]))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/html": "\n    <div class=\"bk-root\">\n        <a href=\"https://bokeh.pydata.org\" target=\"_blank\" class=\"bk-logo bk-logo-small bk-logo-notebook\"></a>\n        <span id=\"1001\">Loading BokehJS ...</span>\n    </div>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "application/javascript": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof (root._bokeh_onload_callbacks) === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  var JS_MIME_TYPE = 'application/javascript';\n  var HTML_MIME_TYPE = 'text/html';\n  var EXEC_MIME_TYPE = 'application/vnd.bokehjs_exec.v0+json';\n  var CLASS_NAME = 'output_bokeh rendered_html';\n\n  /**\n   * Render data to the DOM node\n   */\n  function render(props, node) {\n    var script = document.createElement(\"script\");\n    node.appendChild(script);\n  }\n\n  /**\n   * Handle when an output is cleared or removed\n   */\n  function handleClearOutput(event, handle) {\n    var cell = handle.cell;\n\n    var id = cell.output_area._bokeh_element_id;\n    var server_id = cell.output_area._bokeh_server_id;\n    // Clean up Bokeh references\n    if (id != null && id in Bokeh.index) {\n      Bokeh.index[id].model.document.clear();\n      delete Bokeh.index[id];\n    }\n\n    if (server_id !== undefined) {\n      // Clean up Bokeh references\n      var cmd = \"from bokeh.io.state import curstate; print(curstate().uuid_to_server['\" + server_id + \"'].get_sessions()[0].document.roots[0]._id)\";\n      cell.notebook.kernel.execute(cmd, {\n        iopub: {\n          output: function(msg) {\n            var id = msg.content.text.trim();\n            if (id in Bokeh.index) {\n              Bokeh.index[id].model.document.clear();\n              delete Bokeh.index[id];\n            }\n          }\n        }\n      });\n      // Destroy server and session\n      var cmd = \"import bokeh.io.notebook as ion; ion.destroy_server('\" + server_id + \"')\";\n      cell.notebook.kernel.execute(cmd);\n    }\n  }\n\n  /**\n   * Handle when a new output is added\n   */\n  function handleAddOutput(event, handle) {\n    var output_area = handle.output_area;\n    var output = handle.output;\n\n    // limit handleAddOutput to display_data with EXEC_MIME_TYPE content only\n    if ((output.output_type != \"display_data\") || (!output.data.hasOwnProperty(EXEC_MIME_TYPE))) {\n      return\n    }\n\n    var toinsert = output_area.element.find(\".\" + CLASS_NAME.split(' ')[0]);\n\n    if (output.metadata[EXEC_MIME_TYPE][\"id\"] !== undefined) {\n      toinsert[toinsert.length - 1].firstChild.textContent = output.data[JS_MIME_TYPE];\n      // store reference to embed id on output_area\n      output_area._bokeh_element_id = output.metadata[EXEC_MIME_TYPE][\"id\"];\n    }\n    if (output.metadata[EXEC_MIME_TYPE][\"server_id\"] !== undefined) {\n      var bk_div = document.createElement(\"div\");\n      bk_div.innerHTML = output.data[HTML_MIME_TYPE];\n      var script_attrs = bk_div.children[0].attributes;\n      for (var i = 0; i < script_attrs.length; i++) {\n        toinsert[toinsert.length - 1].firstChild.setAttribute(script_attrs[i].name, script_attrs[i].value);\n      }\n      // store reference to server id on output_area\n      output_area._bokeh_server_id = output.metadata[EXEC_MIME_TYPE][\"server_id\"];\n    }\n  }\n\n  function register_renderer(events, OutputArea) {\n\n    function append_mime(data, metadata, element) {\n      // create a DOM node to render to\n      var toinsert = this.create_output_subarea(\n        metadata,\n        CLASS_NAME,\n        EXEC_MIME_TYPE\n      );\n      this.keyboard_manager.register_events(toinsert);\n      // Render to node\n      var props = {data: data, metadata: metadata[EXEC_MIME_TYPE]};\n      render(props, toinsert[toinsert.length - 1]);\n      element.append(toinsert);\n      return toinsert\n    }\n\n    /* Handle when an output is cleared or removed */\n    events.on('clear_output.CodeCell', handleClearOutput);\n    events.on('delete.Cell', handleClearOutput);\n\n    /* Handle when a new output is added */\n    events.on('output_added.OutputArea', handleAddOutput);\n\n    /**\n     * Register the mime type and append_mime function with output_area\n     */\n    OutputArea.prototype.register_mime_type(EXEC_MIME_TYPE, append_mime, {\n      /* Is output safe? */\n      safe: true,\n      /* Index of renderer in `output_area.display_order` */\n      index: 0\n    });\n  }\n\n  // register the mime type if in Jupyter Notebook environment and previously unregistered\n  if (root.Jupyter !== undefined) {\n    var events = require('base/js/events');\n    var OutputArea = require('notebook/js/outputarea').OutputArea;\n\n    if (OutputArea.prototype.mime_types().indexOf(EXEC_MIME_TYPE) == -1) {\n      register_renderer(events, OutputArea);\n    }\n  }\n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"1001\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) { callback() });\n    }\n    finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.info(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(js_urls, callback) {\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.log(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.log(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = js_urls.length;\n    for (var i = 0; i < js_urls.length; i++) {\n      var url = js_urls[i];\n      var s = document.createElement('script');\n      s.src = url;\n      s.async = false;\n      s.onreadystatechange = s.onload = function() {\n        root._bokeh_is_loading--;\n        if (root._bokeh_is_loading === 0) {\n          console.log(\"Bokeh: all BokehJS libraries loaded\");\n          run_callbacks()\n        }\n      };\n      s.onerror = function() {\n        console.warn(\"failed to load library \" + url);\n      };\n      console.log(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.getElementsByTagName(\"head\")[0].appendChild(s);\n    }\n  };var element = document.getElementById(\"1001\");\n  if (element == null) {\n    console.log(\"Bokeh: ERROR: autoload.js configured with elementid '1001' but no matching script tag was found. \")\n    return false;\n  }\n\n  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-1.0.4.min.js\"];\n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    \n    function(Bokeh) {\n      \n    },\n    function(Bokeh) {\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.css\");\n    }\n  ];\n\n  function run_inline_js() {\n    \n    if ((root.Bokeh !== undefined) || (force === true)) {\n      for (var i = 0; i < inline_js.length; i++) {\n        inline_js[i].call(root, root.Bokeh);\n      }if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"1001\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.log(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(js_urls, function() {\n      console.log(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));",
   "application/vnd.bokehjs_load.v0+json": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof (root._bokeh_onload_callbacks) === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  \n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"1001\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) { callback() });\n    }\n    finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.info(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(js_urls, callback) {\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.log(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.log(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = js_urls.length;\n    for (var i = 0; i < js_urls.length; i++) {\n      var url = js_urls[i];\n      var s = document.createElement('script');\n      s.src = url;\n      s.async = false;\n      s.onreadystatechange = s.onload = function() {\n        root._bokeh_is_loading--;\n        if (root._bokeh_is_loading === 0) {\n          console.log(\"Bokeh: all BokehJS libraries loaded\");\n          run_callbacks()\n        }\n      };\n      s.onerror = function() {\n        console.warn(\"failed to load library \" + url);\n      };\n      console.log(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.getElementsByTagName(\"head\")[0].appendChild(s);\n    }\n  };var element = document.getElementById(\"1001\");\n  if (element == null) {\n    console.log(\"Bokeh: ERROR: autoload.js configured with elementid '1001' but no matching script tag was found. \")\n    return false;\n  }\n\n  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-1.0.4.min.js\"];\n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    \n    function(Bokeh) {\n      \n    },\n    function(Bokeh) {\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.0.4.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.0.4.min.css\");\n    }\n  ];\n\n  function run_inline_js() {\n    \n    if ((root.Bokeh !== undefined) || (force === true)) {\n      for (var i = 0; i < inline_js.length; i++) {\n        inline_js[i].call(root, root.Bokeh);\n      }if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"1001\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.log(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(js_urls, function() {\n      console.log(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

The throughput increases with the batch size in log scale. The device memory, as exepcted, also increases linearly with the batch size. But note that, due to the pooled memory mechanism in MXNet, the measured device memory usage might be different to the actual memory usdage.

One way to measure the actual device memory usage is finding the largest batch size we can run.

```{.python .input  n=3}
# bs = dm.benchmark.load_results('cnn.py__benchmark_largest_batch_size.json')    
# show(plot.max_batch_size(bs))
```

## Throughput on various hardware

```{.python .input  n=4}
# show(plot.throughput_vs_device(thr[(thr.model=='AlexNet')]))
```

```{.python .input  n=5}
# show(plot.throughput_vs_device(thr[(thr.model=='ResNet-v2-50')]))
```

### Prediction accuracy versus throughput

We measture the prediction accuracy of each model using the ILSVRC 2012 validation dataset. Then plot the results together with the throughput with fixed batch size 64. We colorize models from the same family with the same color.

```{.python .input  n=6}
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
```

```{.python .input  n=18}
def split_number(s):
    return s.split('_')
```

```{.python .input  n=19}
paper_results = {
    'resnet50_v1': 0.753,
    'resnet101_v1': 0.764,
    'resnet152_v1': 0.77,
    'resnet18_v1b': 0.6976,
    'resnet34_v1b': 0.733,
    'resnet50_v1b': 0.7615,
    'resnet101_v1b': 0.7737,
    'resnet152_v1b': 0.7831,
    'darknet53': 0.772,
    'inceptionv3': 0.780,
    'mobilenet1.0': 0.709,
    'mobilenet0.75': 0.684,
    'mobilenet0.5': 0.633,
    'mobilenet0.25': 0.498,
}
```

```{.python .input  n=20}
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
    colors = palettes.Category20[max(len(models),3)]
#     colors = palettes.Viridis256[max(len(models),3)]
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
```

```{.python .input  n=21}
def make_dataset(data, model_list):
    data = data.copy()
    data = data[data.index.isin(model_list)]
    assert 'mIoU' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
    colors = palettes.Category20[max(len(models),3)]
#     colors = np.random.choice(palettes.Viridis256, size=max(len(models),3))
#     colors = palettes.Category20b[20] + palettes.Category20c[20]
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
#     data['paper_diff_percent'] = pd.Series(["{0:.2f}%".format((val1-val2) * 100) if val1 > val2 else 'N/A' for val1, val2 in zip(data['accuracy'], data['paper'])], index = data.index)
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
                 ("mIoU", "@mIoU"),
                 ("Improvement over Reference", "@paper_diff_percent"),
                 ("Device memory", "@device_mem MB")]
    
    uniq_model = list(set(data['model_prefix']))
    uniq_model.sort()
    
    legend_map = dict()
    
    for current_prefix in uniq_model:
        indices = [i for i, x in enumerate(data['model_prefix']) if x == current_prefix]
        cc = {k: [v[i] for i in indices] for k, v in data.items()}
        current_source = ColumnDataSource(pd.DataFrame(cc))
        pr = p.circle(x='throughput', y='mIoU', legend='model_prefix', size="size", color='color', source=current_source)
        legend_map[current_prefix] = pr
        
    whisker_map = dict()
#     print(len(p.legend[0].items))
    for ic, row in enumerate(data['model_prefix']):
#         idx = legend_map.index(row)
        if not data['paper'][ic]:
            continue
        w_data = dict(base=[data['throughput'][ic]], lower=[data['paper'][ic]], upper=[data['mIoU'][ic]])
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
    p.yaxis.axis_label = 'mIoU'

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
```

```{.python .input  n=22}
acc = load_results('semantic_segmentation_tesla-v100-sxm2-16gb_accuracy.json')

data = thr[(thr.model.isin(acc.model)) & 
#            (thr.batch_size.isin(acc.batch_size)) &
           (thr.device.isin(acc.device))]
data = data.set_index('model').join(acc[['model','mIoU']].set_index('model'))
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
show(pp)

```

```{.json .output n=22}
[
 {
  "data": {
   "text/html": "\n\n\n\n\n\n  <div class=\"bk-root\" id=\"d1973074-b4d7-427b-9bdd-2e607ac97757\" data-root-id=\"1685\"></div>\n"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "application/javascript": "(function(root) {\n  function embed_document(root) {\n    \n  var docs_json = {\"e0e712a8-bf9b-4511-860a-9cc707fa3c57\":{\"roots\":{\"references\":[{\"attributes\":{\"callback\":null,\"tooltips\":[[\"Model\",\"@model\"],[\"Throughput\",\"@throughput\"],[\"mIoU\",\"@mIoU\"],[\"Improvement over Reference\",\"@paper_diff_percent\"],[\"Device memory\",\"@device_mem MB\"]]},\"id\":\"1767\",\"type\":\"HoverTool\"},{\"attributes\":{\"data_source\":{\"id\":\"1717\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1719\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1720\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1722\",\"type\":\"CDSView\"}},\"id\":\"1721\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1764\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"data_source\":{\"id\":\"1732\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1734\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1735\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1737\",\"type\":\"CDSView\"}},\"id\":\"1736\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"source\":{\"id\":\"1748\",\"type\":\"ColumnDataSource\"}},\"id\":\"1753\",\"type\":\"CDSView\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1734\",\"type\":\"Circle\"},{\"attributes\":{},\"id\":\"1727\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"click_policy\":\"hide\",\"items\":[{\"id\":\"1731\",\"type\":\"LegendItem\"},{\"id\":\"1747\",\"type\":\"LegendItem\"},{\"id\":\"1765\",\"type\":\"LegendItem\"}],\"label_text_font_size\":{\"value\":\"1em\"},\"location\":\"bottom_left\",\"plot\":{\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"}},\"id\":\"1730\",\"type\":\"Legend\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1752\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1765\",\"type\":\"LegendItem\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1719\",\"type\":\"Circle\"},{\"attributes\":{\"num_minor_ticks\":10},\"id\":\"1695\",\"type\":\"LogTicker\"},{\"attributes\":{},\"id\":\"1700\",\"type\":\"BasicTicker\"},{\"attributes\":{\"active_drag\":{\"id\":\"1704\",\"type\":\"PanTool\"},\"active_inspect\":\"auto\",\"active_multi\":null,\"active_scroll\":\"auto\",\"active_tap\":\"auto\",\"logo\":null,\"tools\":[{\"id\":\"1704\",\"type\":\"PanTool\"},{\"id\":\"1705\",\"type\":\"WheelZoomTool\"},{\"id\":\"1706\",\"type\":\"BoxZoomTool\"},{\"id\":\"1707\",\"type\":\"SaveTool\"},{\"id\":\"1708\",\"type\":\"ResetTool\"},{\"id\":\"1767\",\"type\":\"HoverTool\"}]},\"id\":\"1709\",\"type\":\"Toolbar\"},{\"attributes\":{},\"id\":\"1763\",\"type\":\"Selection\"},{\"attributes\":{\"plot\":null,\"text\":\"\",\"text_font_size\":{\"value\":\"100%\"}},\"id\":\"1723\",\"type\":\"Title\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1735\",\"type\":\"Circle\"},{\"attributes\":{\"source\":{\"id\":\"1717\",\"type\":\"ColumnDataSource\"}},\"id\":\"1722\",\"type\":\"CDSView\"},{\"attributes\":{\"overlay\":{\"id\":\"1711\",\"type\":\"BoxAnnotation\"}},\"id\":\"1706\",\"type\":\"BoxZoomTool\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"below\":[{\"id\":\"1694\",\"type\":\"LogAxis\"}],\"border_fill_alpha\":{\"value\":0},\"left\":[{\"id\":\"1699\",\"type\":\"LinearAxis\"}],\"renderers\":[{\"id\":\"1694\",\"type\":\"LogAxis\"},{\"id\":\"1698\",\"type\":\"Grid\"},{\"id\":\"1699\",\"type\":\"LinearAxis\"},{\"id\":\"1703\",\"type\":\"Grid\"},{\"id\":\"1711\",\"type\":\"BoxAnnotation\"},{\"id\":\"1730\",\"type\":\"Legend\"},{\"id\":\"1721\",\"type\":\"GlyphRenderer\"},{\"id\":\"1736\",\"type\":\"GlyphRenderer\"},{\"id\":\"1752\",\"type\":\"GlyphRenderer\"}],\"sizing_mode\":\"scale_width\",\"title\":{\"id\":\"1723\",\"type\":\"Title\"},\"toolbar\":{\"id\":\"1709\",\"type\":\"Toolbar\"},\"toolbar_location\":\"above\",\"x_range\":{\"id\":\"1686\",\"type\":\"DataRange1d\"},\"x_scale\":{\"id\":\"1690\",\"type\":\"LogScale\"},\"y_range\":{\"id\":\"1688\",\"type\":\"DataRange1d\"},\"y_scale\":{\"id\":\"1692\",\"type\":\"LinearScale\"}},\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#aec7e8\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[7500],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"xqDlMFMx5T8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"fcn_resnet101_coco\"],\"model_prefix\":[\"fcn\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"aPplhakaPEA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"+FB2LCBEQEA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1763\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1764\",\"type\":\"UnionRenderers\"}},\"id\":\"1732\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1720\",\"type\":\"Circle\"},{\"attributes\":{},\"id\":\"1704\",\"type\":\"PanTool\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#ff7f0e\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[8546],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"re80TZti5j8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"psp_resnet101_coco\"],\"model_prefix\":[\"psp\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"AAAAAAAAPkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"ecCYWPOSOkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1811\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1812\",\"type\":\"UnionRenderers\"}},\"id\":\"1748\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"data_source\":{\"id\":\"1748\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1750\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1751\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1753\",\"type\":\"CDSView\"}},\"id\":\"1752\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1705\",\"type\":\"WheelZoomTool\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1721\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1731\",\"type\":\"LegendItem\"},{\"attributes\":{\"axis_label\":\"mIoU\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1727\",\"type\":\"BasicTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1700\",\"type\":\"BasicTicker\"}},\"id\":\"1699\",\"type\":\"LinearAxis\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1750\",\"type\":\"Circle\"},{\"attributes\":{},\"id\":\"1746\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1736\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1747\",\"type\":\"LegendItem\"},{\"attributes\":{\"dimension\":1,\"plot\":{\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1700\",\"type\":\"BasicTicker\"}},\"id\":\"1703\",\"type\":\"Grid\"},{\"attributes\":{\"callback\":null},\"id\":\"1688\",\"type\":\"DataRange1d\"},{\"attributes\":{},\"id\":\"1692\",\"type\":\"LinearScale\"},{\"attributes\":{},\"id\":\"1811\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1707\",\"type\":\"SaveTool\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1751\",\"type\":\"Circle\"},{\"attributes\":{\"ticker\":null},\"id\":\"1725\",\"type\":\"LogTickFormatter\"},{\"attributes\":{\"source\":{\"id\":\"1732\",\"type\":\"ColumnDataSource\"}},\"id\":\"1737\",\"type\":\"CDSView\"},{\"attributes\":{\"grid_line_color\":{\"value\":null},\"plot\":{\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1695\",\"type\":\"LogTicker\"}},\"id\":\"1698\",\"type\":\"Grid\"},{\"attributes\":{},\"id\":\"1812\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"bottom_units\":\"screen\",\"fill_alpha\":{\"value\":0.5},\"fill_color\":{\"value\":\"lightgrey\"},\"left_units\":\"screen\",\"level\":\"overlay\",\"line_alpha\":{\"value\":1.0},\"line_color\":{\"value\":\"black\"},\"line_dash\":[4,4],\"line_width\":{\"value\":2},\"plot\":null,\"render_mode\":\"css\",\"right_units\":\"screen\",\"top_units\":\"screen\"},\"id\":\"1711\",\"type\":\"BoxAnnotation\"},{\"attributes\":{\"callback\":null},\"id\":\"1686\",\"type\":\"DataRange1d\"},{\"attributes\":{},\"id\":\"1745\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1690\",\"type\":\"LogScale\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32,32],\"color\":[\"#1f77b4\",\"#1f77b4\"],\"device\":[\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\"],\"device_mem\":[6620,6576],\"index\":[0,1],\"mIoU\":{\"__ndarray__\":\"XVhKTrSJ5j+JTaE/7OHmPw==\",\"dtype\":\"float64\",\"shape\":[2]},\"model\":[\"deeplab_resnet101_coco\",\"deeplab_resnet152_coco\"],\"model_prefix\":[\"deeplab\",\"deeplab\"],\"paper\":[0,0],\"size\":{\"__ndarray__\":\"J+T+4WhnOkDDBiax6FA6QA==\",\"dtype\":\"float64\",\"shape\":[2]},\"throughput\":{\"__ndarray__\":\"jBLOnevNOUCuHMVBGF40QA==\",\"dtype\":\"float64\",\"shape\":[2]},\"workload\":[\"Inference\",\"Inference\"]},\"selected\":{\"id\":\"1745\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1746\",\"type\":\"UnionRenderers\"}},\"id\":\"1717\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"axis_label\":\"#samples/sec\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1725\",\"type\":\"LogTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1685\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1695\",\"type\":\"LogTicker\"}},\"id\":\"1694\",\"type\":\"LogAxis\"},{\"attributes\":{},\"id\":\"1708\",\"type\":\"ResetTool\"}],\"root_ids\":[\"1685\"]},\"title\":\"Bokeh Application\",\"version\":\"1.0.4\"}};\n  var render_items = [{\"docid\":\"e0e712a8-bf9b-4511-860a-9cc707fa3c57\",\"roots\":{\"1685\":\"d1973074-b4d7-427b-9bdd-2e607ac97757\"}}];\n  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);\n\n  }\n  if (root.Bokeh !== undefined) {\n    embed_document(root);\n  } else {\n    var attempts = 0;\n    var timer = setInterval(function(root) {\n      if (root.Bokeh !== undefined) {\n        embed_document(root);\n        clearInterval(timer);\n      }\n      attempts++;\n      if (attempts > 100) {\n        console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n        clearInterval(timer);\n      }\n    }, 10, root)\n  }\n})(window);",
   "application/vnd.bokehjs_exec.v0+json": ""
  },
  "metadata": {
   "application/vnd.bokehjs_exec.v0+json": {
    "id": "1685"
   }
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=16}
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
import os

html = file_html(pp, CDN, "semantic_segmentation")
html = html.replace('Bokeh.set_log_level("info");', '''Bokeh.set_log_level("info"); window.onload = function () { window.dispatchEvent(new Event('resize')); };''')
print(html)
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n\n\n\n<!DOCTYPE html>\n<html lang=\"en\">\n  \n  <head>\n    \n      <meta charset=\"utf-8\">\n      <title>semantic_segmentation</title>\n      \n      \n        \n          \n        <link rel=\"stylesheet\" href=\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\" type=\"text/css\" />\n        \n        \n          \n        <script type=\"text/javascript\" src=\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js\"></script>\n        <script type=\"text/javascript\">\n            Bokeh.set_log_level(\"info\"); window.onload = function () { window.dispatchEvent(new Event('resize')); };\n        </script>\n        \n      \n      \n    \n  </head>\n  \n  \n  <body>\n    \n      \n        \n          \n          \n            \n              <div class=\"bk-root\" id=\"6c8453ac-9914-4b21-8084-5cc1692e234b\" data-root-id=\"1322\"></div>\n            \n          \n        \n      \n      \n        <script type=\"application/json\" id=\"1667\">\n          {\"4bb88869-bb9c-4331-af0a-233b7bba9db3\":{\"roots\":{\"references\":[{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1356\",\"type\":\"Circle\"},{\"attributes\":{\"data_source\":{\"id\":\"1354\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1356\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1357\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1359\",\"type\":\"CDSView\"}},\"id\":\"1358\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"num_minor_ticks\":10},\"id\":\"1332\",\"type\":\"LogTicker\"},{\"attributes\":{\"source\":{\"id\":\"1369\",\"type\":\"ColumnDataSource\"}},\"id\":\"1374\",\"type\":\"CDSView\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1372\",\"type\":\"Circle\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#1f77b4\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[6620],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"XVhKTrSJ5j8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"deeplab_resnet101_coco\"],\"model_prefix\":[\"deeplab_resnet101_coco\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"J+T+4WhnOkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"jBLOnevNOUA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1382\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1383\",\"type\":\"UnionRenderers\"}},\"id\":\"1354\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"bottom_units\":\"screen\",\"fill_alpha\":{\"value\":0.5},\"fill_color\":{\"value\":\"lightgrey\"},\"left_units\":\"screen\",\"level\":\"overlay\",\"line_alpha\":{\"value\":1.0},\"line_color\":{\"value\":\"black\"},\"line_dash\":[4,4],\"line_width\":{\"value\":2},\"plot\":null,\"render_mode\":\"css\",\"right_units\":\"screen\",\"top_units\":\"screen\"},\"id\":\"1348\",\"type\":\"BoxAnnotation\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1389\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1402\",\"type\":\"LegendItem\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#ff7f0e\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[7500],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"xqDlMFMx5T8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"fcn_resnet101_coco\"],\"model_prefix\":[\"fcn_resnet101_coco\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"aPplhakaPEA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"+FB2LCBEQEA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1420\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1421\",\"type\":\"UnionRenderers\"}},\"id\":\"1385\",\"type\":\"ColumnDataSource\"},{\"attributes\":{},\"id\":\"1327\",\"type\":\"LogScale\"},{\"attributes\":{},\"id\":\"1400\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1420\",\"type\":\"Selection\"},{\"attributes\":{\"source\":{\"id\":\"1385\",\"type\":\"ColumnDataSource\"}},\"id\":\"1390\",\"type\":\"CDSView\"},{\"attributes\":{\"data_source\":{\"id\":\"1369\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1371\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1372\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1374\",\"type\":\"CDSView\"}},\"id\":\"1373\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"axis_label\":\"#samples/sec\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1362\",\"type\":\"LogTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1332\",\"type\":\"LogTicker\"}},\"id\":\"1331\",\"type\":\"LogAxis\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1373\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1384\",\"type\":\"LegendItem\"},{\"attributes\":{\"callback\":null,\"tooltips\":[[\"Model\",\"@model\"],[\"Throughput\",\"@throughput\"],[\"mIoU\",\"@mIoU\"],[\"Improvement over Reference\",\"@paper_diff_percent\"],[\"Device memory\",\"@device_mem MB\"]]},\"id\":\"1424\",\"type\":\"HoverTool\"},{\"attributes\":{\"axis_label\":\"mIoU\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1364\",\"type\":\"BasicTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1337\",\"type\":\"BasicTicker\"}},\"id\":\"1336\",\"type\":\"LinearAxis\"},{\"attributes\":{\"callback\":null},\"id\":\"1325\",\"type\":\"DataRange1d\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"below\":[{\"id\":\"1331\",\"type\":\"LogAxis\"}],\"border_fill_alpha\":{\"value\":0},\"left\":[{\"id\":\"1336\",\"type\":\"LinearAxis\"}],\"renderers\":[{\"id\":\"1331\",\"type\":\"LogAxis\"},{\"id\":\"1335\",\"type\":\"Grid\"},{\"id\":\"1336\",\"type\":\"LinearAxis\"},{\"id\":\"1340\",\"type\":\"Grid\"},{\"id\":\"1348\",\"type\":\"BoxAnnotation\"},{\"id\":\"1367\",\"type\":\"Legend\"},{\"id\":\"1358\",\"type\":\"GlyphRenderer\"},{\"id\":\"1373\",\"type\":\"GlyphRenderer\"},{\"id\":\"1389\",\"type\":\"GlyphRenderer\"},{\"id\":\"1407\",\"type\":\"GlyphRenderer\"}],\"sizing_mode\":\"scale_width\",\"title\":{\"id\":\"1360\",\"type\":\"Title\"},\"toolbar\":{\"id\":\"1346\",\"type\":\"Toolbar\"},\"toolbar_location\":\"above\",\"x_range\":{\"id\":\"1323\",\"type\":\"DataRange1d\"},\"x_scale\":{\"id\":\"1327\",\"type\":\"LogScale\"},\"y_range\":{\"id\":\"1325\",\"type\":\"DataRange1d\"},\"y_scale\":{\"id\":\"1329\",\"type\":\"LinearScale\"}},\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{},\"id\":\"1341\",\"type\":\"PanTool\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1387\",\"type\":\"Circle\"},{\"attributes\":{},\"id\":\"1401\",\"type\":\"UnionRenderers\"},{\"attributes\":{},\"id\":\"1337\",\"type\":\"BasicTicker\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1407\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1422\",\"type\":\"LegendItem\"},{\"attributes\":{\"source\":{\"id\":\"1403\",\"type\":\"ColumnDataSource\"}},\"id\":\"1408\",\"type\":\"CDSView\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#ffbb78\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[8546],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"re80TZti5j8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"psp_resnet101_coco\"],\"model_prefix\":[\"psp_resnet101_coco\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"AAAAAAAAPkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"ecCYWPOSOkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1455\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1456\",\"type\":\"UnionRenderers\"}},\"id\":\"1403\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1405\",\"type\":\"Circle\"},{\"attributes\":{\"callback\":null},\"id\":\"1323\",\"type\":\"DataRange1d\"},{\"attributes\":{\"grid_line_color\":{\"value\":null},\"plot\":{\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1332\",\"type\":\"LogTicker\"}},\"id\":\"1335\",\"type\":\"Grid\"},{\"attributes\":{\"active_drag\":{\"id\":\"1341\",\"type\":\"PanTool\"},\"active_inspect\":\"auto\",\"active_multi\":null,\"active_scroll\":\"auto\",\"active_tap\":\"auto\",\"logo\":null,\"tools\":[{\"id\":\"1341\",\"type\":\"PanTool\"},{\"id\":\"1342\",\"type\":\"WheelZoomTool\"},{\"id\":\"1343\",\"type\":\"BoxZoomTool\"},{\"id\":\"1344\",\"type\":\"SaveTool\"},{\"id\":\"1345\",\"type\":\"ResetTool\"},{\"id\":\"1424\",\"type\":\"HoverTool\"}]},\"id\":\"1346\",\"type\":\"Toolbar\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1371\",\"type\":\"Circle\"},{\"attributes\":{},\"id\":\"1364\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1357\",\"type\":\"Circle\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1406\",\"type\":\"Circle\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1358\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1368\",\"type\":\"LegendItem\"},{\"attributes\":{\"dimension\":1,\"plot\":{\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1337\",\"type\":\"BasicTicker\"}},\"id\":\"1340\",\"type\":\"Grid\"},{\"attributes\":{\"data_source\":{\"id\":\"1403\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1405\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1406\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1408\",\"type\":\"CDSView\"}},\"id\":\"1407\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"source\":{\"id\":\"1354\",\"type\":\"ColumnDataSource\"}},\"id\":\"1359\",\"type\":\"CDSView\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"click_policy\":\"hide\",\"items\":[{\"id\":\"1368\",\"type\":\"LegendItem\"},{\"id\":\"1384\",\"type\":\"LegendItem\"},{\"id\":\"1402\",\"type\":\"LegendItem\"},{\"id\":\"1422\",\"type\":\"LegendItem\"}],\"label_text_font_size\":{\"value\":\"1em\"},\"location\":\"bottom_left\",\"plot\":{\"id\":\"1322\",\"subtype\":\"Figure\",\"type\":\"Plot\"}},\"id\":\"1367\",\"type\":\"Legend\"},{\"attributes\":{},\"id\":\"1345\",\"type\":\"ResetTool\"},{\"attributes\":{},\"id\":\"1421\",\"type\":\"UnionRenderers\"},{\"attributes\":{},\"id\":\"1382\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1383\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"callback\":null,\"data\":{\"batch_size\":[32],\"color\":[\"#aec7e8\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[6576],\"index\":[0],\"mIoU\":{\"__ndarray__\":\"iU2hP+zh5j8=\",\"dtype\":\"float64\",\"shape\":[1]},\"model\":[\"deeplab_resnet152_coco\"],\"model_prefix\":[\"deeplab_resnet152_coco\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"wwYmsehQOkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"rhzFQRheNEA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1400\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1401\",\"type\":\"UnionRenderers\"}},\"id\":\"1369\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"plot\":null,\"text\":\"\",\"text_font_size\":{\"value\":\"100%\"}},\"id\":\"1360\",\"type\":\"Title\"},{\"attributes\":{\"ticker\":null},\"id\":\"1362\",\"type\":\"LogTickFormatter\"},{\"attributes\":{\"data_source\":{\"id\":\"1385\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1387\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1388\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1390\",\"type\":\"CDSView\"}},\"id\":\"1389\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1456\",\"type\":\"UnionRenderers\"},{\"attributes\":{},\"id\":\"1344\",\"type\":\"SaveTool\"},{\"attributes\":{},\"id\":\"1329\",\"type\":\"LinearScale\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"mIoU\"}},\"id\":\"1388\",\"type\":\"Circle\"},{\"attributes\":{\"overlay\":{\"id\":\"1348\",\"type\":\"BoxAnnotation\"}},\"id\":\"1343\",\"type\":\"BoxZoomTool\"},{\"attributes\":{},\"id\":\"1455\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1342\",\"type\":\"WheelZoomTool\"}],\"root_ids\":[\"1322\"]},\"title\":\"Bokeh Application\",\"version\":\"1.0.4\"}}\n        </script>\n        <script type=\"text/javascript\">\n          (function() {\n            var fn = function() {\n              Bokeh.safely(function() {\n                (function(root) {\n                  function embed_document(root) {\n                    \n                  var docs_json = document.getElementById('1667').textContent;\n                  var render_items = [{\"docid\":\"4bb88869-bb9c-4331-af0a-233b7bba9db3\",\"roots\":{\"1322\":\"6c8453ac-9914-4b21-8084-5cc1692e234b\"}}];\n                  root.Bokeh.embed.embed_items(docs_json, render_items);\n                \n                  }\n                  if (root.Bokeh !== undefined) {\n                    embed_document(root);\n                  } else {\n                    var attempts = 0;\n                    var timer = setInterval(function(root) {\n                      if (root.Bokeh !== undefined) {\n                        embed_document(root);\n                        clearInterval(timer);\n                      }\n                      attempts++;\n                      if (attempts > 100) {\n                        console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n                        clearInterval(timer);\n                      }\n                    }, 10, root)\n                  }\n                })(window);\n              });\n            };\n            if (document.readyState != \"loading\") fn();\n            else document.addEventListener(\"DOMContentLoaded\", fn);\n          })();\n        </script>\n    \n  </body>\n  \n</html>\n"
 }
]
```

```{.python .input  n=17}
with open(os.path.join(os.path.abspath(''), 'semantic_segmentation_throughputs.html'), 'wt') as f:
    f.write(html)
```

```{.python .input}

```