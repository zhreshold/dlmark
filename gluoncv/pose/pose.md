# Convolutional Neural Network

We benchmark the convolutional neural networks provided by the [Gluon modelzoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html).

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=1}
import dlmark as dm

thr = dm.benchmark.load_results('pose.py__benchmark_throughput*json')

models = thr.model.unique()
devices = thr.device.unique()
(models, devices)
```

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "(array(['alpha_pose_resnet101_v1b_coco', 'simple_pose_resnet101_v1b',\n        'simple_pose_resnet101_v1d', 'simple_pose_resnet152_v1b',\n        'simple_pose_resnet152_v1d', 'simple_pose_resnet18_v1b',\n        'simple_pose_resnet50_v1b', 'simple_pose_resnet50_v1d'],\n       dtype=object), array(['Tesla V100-SXM2-16GB'], dtype=object))"
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

```{.python .input  n=7}
def split_number(s):
    return s.split('_')
```

```{.python .input  n=8}
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

```{.python .input  n=33}
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
                 ("AP", "@accuracy"),
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

```{.python .input  n=37}
def make_dataset(data, model_list):
    data = data.copy()
    data = data[data.index.isin(model_list)]
    assert 'accuracy' in data.columns, data.columns
    assert ('model' in data.columns or
            'model_prefix' in data.columns), data.columns
    model = 'model_prefix' if 'model_prefix' in data.columns else 'model'
    models = sorted(data[model].unique())
    colors = palettes.Category10[10]
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
                 ("AP", "@accuracy"),
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
    p.yaxis.axis_label = 'AP'

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

```{.python .input  n=38}
acc = load_results('pose_tesla-v100-sxm2-16gb_accuracy.json')

data = thr[(thr.model.isin(acc.model)) & 
#            (thr.batch_size.isin(acc.batch_size)) &
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
show(pp)

```

```{.json .output n=38}
[
 {
  "data": {
   "text/html": "\n\n\n\n\n\n  <div class=\"bk-root\" id=\"f3536833-9cc1-4ecd-b941-49312cc3bb12\" data-root-id=\"2439\"></div>\n"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "application/javascript": "(function(root) {\n  function embed_document(root) {\n    \n  var docs_json = {\"3fd55391-18ab-4ec2-8653-97f6a6ed9efa\":{\"roots\":{\"references\":[{\"attributes\":{\"callback\":null},\"id\":\"2442\",\"type\":\"DataRange1d\"},{\"attributes\":{\"bottom_units\":\"screen\",\"fill_alpha\":{\"value\":0.5},\"fill_color\":{\"value\":\"lightgrey\"},\"left_units\":\"screen\",\"level\":\"overlay\",\"line_alpha\":{\"value\":1.0},\"line_color\":{\"value\":\"black\"},\"line_dash\":[4,4],\"line_width\":{\"value\":2},\"plot\":null,\"render_mode\":\"css\",\"right_units\":\"screen\",\"top_units\":\"screen\"},\"id\":\"2465\",\"type\":\"BoxAnnotation\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"2488\",\"type\":\"Circle\"},{\"attributes\":{\"grid_line_color\":{\"value\":null},\"plot\":{\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"2449\",\"type\":\"LogTicker\"}},\"id\":\"2452\",\"type\":\"Grid\"},{\"attributes\":{\"ticker\":null},\"id\":\"2479\",\"type\":\"LogTickFormatter\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"below\":[{\"id\":\"2448\",\"type\":\"LogAxis\"}],\"border_fill_alpha\":{\"value\":0},\"left\":[{\"id\":\"2453\",\"type\":\"LinearAxis\"}],\"renderers\":[{\"id\":\"2448\",\"type\":\"LogAxis\"},{\"id\":\"2452\",\"type\":\"Grid\"},{\"id\":\"2453\",\"type\":\"LinearAxis\"},{\"id\":\"2457\",\"type\":\"Grid\"},{\"id\":\"2465\",\"type\":\"BoxAnnotation\"},{\"id\":\"2484\",\"type\":\"Legend\"},{\"id\":\"2475\",\"type\":\"GlyphRenderer\"},{\"id\":\"2490\",\"type\":\"GlyphRenderer\"}],\"sizing_mode\":\"scale_width\",\"title\":{\"id\":\"2478\",\"type\":\"Title\"},\"toolbar\":{\"id\":\"2463\",\"type\":\"Toolbar\"},\"toolbar_location\":\"above\",\"x_range\":{\"id\":\"2440\",\"type\":\"DataRange1d\"},\"x_scale\":{\"id\":\"2444\",\"type\":\"LogScale\"},\"y_range\":{\"id\":\"2442\",\"type\":\"DataRange1d\"},\"y_scale\":{\"id\":\"2446\",\"type\":\"LinearScale\"}},\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"2490\",\"type\":\"GlyphRenderer\"}]},\"id\":\"2501\",\"type\":\"LegendItem\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"2489\",\"type\":\"Circle\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"click_policy\":\"hide\",\"items\":[{\"id\":\"2485\",\"type\":\"LegendItem\"},{\"id\":\"2501\",\"type\":\"LegendItem\"}],\"label_text_font_size\":{\"value\":\"1em\"},\"location\":\"bottom_left\",\"plot\":{\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"}},\"id\":\"2484\",\"type\":\"Legend\"},{\"attributes\":{},\"id\":\"2459\",\"type\":\"WheelZoomTool\"},{\"attributes\":{\"callback\":null,\"data\":{\"accuracy\":{\"__ndarray__\":\"tDgU74S65z8=\",\"dtype\":\"float64\",\"shape\":[1]},\"batch_size\":[64],\"color\":[\"#1f77b4\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[1568],\"index\":[0],\"model\":[\"alpha_pose_resnet101_v1b_coco\"],\"model_prefix\":[\"alpha\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"AAAAAAAAPkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"Rc2INYKQeUA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"2498\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"2499\",\"type\":\"UnionRenderers\"}},\"id\":\"2471\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"overlay\":{\"id\":\"2465\",\"type\":\"BoxAnnotation\"}},\"id\":\"2460\",\"type\":\"BoxZoomTool\"},{\"attributes\":{\"source\":{\"id\":\"2471\",\"type\":\"ColumnDataSource\"}},\"id\":\"2476\",\"type\":\"CDSView\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"2474\",\"type\":\"Circle\"},{\"attributes\":{\"plot\":null,\"text\":\"\",\"text_font_size\":{\"value\":\"100%\"}},\"id\":\"2478\",\"type\":\"Title\"},{\"attributes\":{},\"id\":\"2481\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"2592\",\"type\":\"UnionRenderers\"},{\"attributes\":{},\"id\":\"2454\",\"type\":\"BasicTicker\"},{\"attributes\":{\"callback\":null,\"data\":{\"accuracy\":{\"__ndarray__\":\"wMpOKpQu5z+pbvAqP1nnP1P0KWaFLuc/R1EJn5CA5z9x7kq24DflPwGuXOw9uuY/r/9Aombx5j8=\",\"dtype\":\"float64\",\"shape\":[7]},\"batch_size\":[64,64,64,64,64,64,64],\"color\":[\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\"],\"device\":[\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\"],\"device_mem\":[690,722,674,694,578,714,690],\"index\":[0,1,2,3,4,5,6],\"model\":[\"simple_pose_resnet101_v1b\",\"simple_pose_resnet101_v1d\",\"simple_pose_resnet152_v1b\",\"simple_pose_resnet152_v1d\",\"simple_pose_resnet18_v1b\",\"simple_pose_resnet50_v1b\",\"simple_pose_resnet50_v1d\"],\"model_prefix\":[\"simple\",\"simple\",\"simple\",\"simple\",\"simple\",\"simple\",\"simple\"],\"paper\":[0,0,0,0,0,0,0],\"size\":{\"__ndarray__\":\"10QEi6HmM0C3bdu2bVs0QHH1QGU3qzNA1nmncWD1M0Bt27Zt2zYyQDj779h5PjRA10QEi6HmM0A=\",\"dtype\":\"float64\",\"shape\":[7]},\"throughput\":{\"__ndarray__\":\"SMgNkCkVgUAg23L/JHGAQIKVtDkEb3lAz5kaNw1feECxkyoO/7SdQC/9Wi6iLolAKMwoCxPSh0A=\",\"dtype\":\"float64\",\"shape\":[7]},\"workload\":[\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\"]},\"selected\":{\"id\":\"2591\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"2592\",\"type\":\"UnionRenderers\"}},\"id\":\"2486\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"callback\":null,\"tooltips\":[[\"Model\",\"@model\"],[\"Throughput\",\"@throughput\"],[\"AP\",\"@accuracy\"],[\"Improvement over Reference\",\"@paper_diff_percent\"],[\"Device memory\",\"@device_mem MB\"]]},\"id\":\"2503\",\"type\":\"HoverTool\"},{\"attributes\":{\"data_source\":{\"id\":\"2471\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"2473\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"2474\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"2476\",\"type\":\"CDSView\"}},\"id\":\"2475\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"2473\",\"type\":\"Circle\"},{\"attributes\":{\"active_drag\":{\"id\":\"2458\",\"type\":\"PanTool\"},\"active_inspect\":\"auto\",\"active_multi\":null,\"active_scroll\":\"auto\",\"active_tap\":\"auto\",\"logo\":null,\"tools\":[{\"id\":\"2458\",\"type\":\"PanTool\"},{\"id\":\"2459\",\"type\":\"WheelZoomTool\"},{\"id\":\"2460\",\"type\":\"BoxZoomTool\"},{\"id\":\"2461\",\"type\":\"SaveTool\"},{\"id\":\"2462\",\"type\":\"ResetTool\"},{\"id\":\"2503\",\"type\":\"HoverTool\"}]},\"id\":\"2463\",\"type\":\"Toolbar\"},{\"attributes\":{},\"id\":\"2591\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"2462\",\"type\":\"ResetTool\"},{\"attributes\":{},\"id\":\"2499\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"axis_label\":\"#samples/sec\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"2479\",\"type\":\"LogTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"2449\",\"type\":\"LogTicker\"}},\"id\":\"2448\",\"type\":\"LogAxis\"},{\"attributes\":{\"num_minor_ticks\":10},\"id\":\"2449\",\"type\":\"LogTicker\"},{\"attributes\":{\"callback\":null},\"id\":\"2440\",\"type\":\"DataRange1d\"},{\"attributes\":{},\"id\":\"2461\",\"type\":\"SaveTool\"},{\"attributes\":{},\"id\":\"2458\",\"type\":\"PanTool\"},{\"attributes\":{\"axis_label\":\"AP\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"2481\",\"type\":\"BasicTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"2454\",\"type\":\"BasicTicker\"}},\"id\":\"2453\",\"type\":\"LinearAxis\"},{\"attributes\":{\"data_source\":{\"id\":\"2486\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"2488\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"2489\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"2491\",\"type\":\"CDSView\"}},\"id\":\"2490\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"2498\",\"type\":\"Selection\"},{\"attributes\":{\"dimension\":1,\"plot\":{\"id\":\"2439\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"2454\",\"type\":\"BasicTicker\"}},\"id\":\"2457\",\"type\":\"Grid\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"2475\",\"type\":\"GlyphRenderer\"}]},\"id\":\"2485\",\"type\":\"LegendItem\"},{\"attributes\":{},\"id\":\"2446\",\"type\":\"LinearScale\"},{\"attributes\":{\"source\":{\"id\":\"2486\",\"type\":\"ColumnDataSource\"}},\"id\":\"2491\",\"type\":\"CDSView\"},{\"attributes\":{},\"id\":\"2444\",\"type\":\"LogScale\"}],\"root_ids\":[\"2439\"]},\"title\":\"Bokeh Application\",\"version\":\"1.0.4\"}};\n  var render_items = [{\"docid\":\"3fd55391-18ab-4ec2-8653-97f6a6ed9efa\",\"roots\":{\"2439\":\"f3536833-9cc1-4ecd-b941-49312cc3bb12\"}}];\n  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);\n\n  }\n  if (root.Bokeh !== undefined) {\n    embed_document(root);\n  } else {\n    var attempts = 0;\n    var timer = setInterval(function(root) {\n      if (root.Bokeh !== undefined) {\n        embed_document(root);\n        clearInterval(timer);\n      }\n      attempts++;\n      if (attempts > 100) {\n        console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n        clearInterval(timer);\n      }\n    }, 10, root)\n  }\n})(window);",
   "application/vnd.bokehjs_exec.v0+json": ""
  },
  "metadata": {
   "application/vnd.bokehjs_exec.v0+json": {
    "id": "2439"
   }
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=32}
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
import os

html = file_html(pp, CDN, "pose")
html = html.replace('Bokeh.set_log_level("info");', '''Bokeh.set_log_level("info"); window.onload = function () { window.dispatchEvent(new Event('resize')); };''')
print(html)
```

```{.json .output n=32}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n\n\n\n<!DOCTYPE html>\n<html lang=\"en\">\n  \n  <head>\n    \n      <meta charset=\"utf-8\">\n      <title>pose</title>\n      \n      \n        \n          \n        <link rel=\"stylesheet\" href=\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css\" type=\"text/css\" />\n        \n        \n          \n        <script type=\"text/javascript\" src=\"https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js\"></script>\n        <script type=\"text/javascript\">\n            Bokeh.set_log_level(\"info\"); window.onload = function () { window.dispatchEvent(new Event('resize')); };\n        </script>\n        \n      \n      \n    \n  </head>\n  \n  \n  <body>\n    \n      \n        \n          \n          \n            \n              <div class=\"bk-root\" id=\"808c0fe4-add6-40ea-b556-ed10866672e8\" data-root-id=\"1926\"></div>\n            \n          \n        \n      \n      \n        <script type=\"application/json\" id=\"2213\">\n          {\"1b922361-77d4-4c72-9f1a-331d591c2315\":{\"roots\":{\"references\":[{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"1976\",\"type\":\"Circle\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1977\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1988\",\"type\":\"LegendItem\"},{\"attributes\":{},\"id\":\"1985\",\"type\":\"Selection\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"click_policy\":\"hide\",\"items\":[{\"id\":\"1972\",\"type\":\"LegendItem\"},{\"id\":\"1988\",\"type\":\"LegendItem\"}],\"label_text_font_size\":{\"value\":\"1em\"},\"location\":\"bottom_left\",\"plot\":{\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"}},\"id\":\"1971\",\"type\":\"Legend\"},{\"attributes\":{},\"id\":\"1986\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"bottom_units\":\"screen\",\"fill_alpha\":{\"value\":0.5},\"fill_color\":{\"value\":\"lightgrey\"},\"left_units\":\"screen\",\"level\":\"overlay\",\"line_alpha\":{\"value\":1.0},\"line_color\":{\"value\":\"black\"},\"line_dash\":[4,4],\"line_width\":{\"value\":2},\"plot\":null,\"render_mode\":\"css\",\"right_units\":\"screen\",\"top_units\":\"screen\"},\"id\":\"1952\",\"type\":\"BoxAnnotation\"},{\"attributes\":{},\"id\":\"1948\",\"type\":\"SaveTool\"},{\"attributes\":{},\"id\":\"1946\",\"type\":\"WheelZoomTool\"},{\"attributes\":{\"data_source\":{\"id\":\"1958\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1960\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1961\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1963\",\"type\":\"CDSView\"}},\"id\":\"1962\",\"type\":\"GlyphRenderer\"},{\"attributes\":{\"active_drag\":{\"id\":\"1945\",\"type\":\"PanTool\"},\"active_inspect\":\"auto\",\"active_multi\":null,\"active_scroll\":\"auto\",\"active_tap\":\"auto\",\"logo\":null,\"tools\":[{\"id\":\"1945\",\"type\":\"PanTool\"},{\"id\":\"1946\",\"type\":\"WheelZoomTool\"},{\"id\":\"1947\",\"type\":\"BoxZoomTool\"},{\"id\":\"1948\",\"type\":\"SaveTool\"},{\"id\":\"1949\",\"type\":\"ResetTool\"},{\"id\":\"1990\",\"type\":\"HoverTool\"}]},\"id\":\"1950\",\"type\":\"Toolbar\"},{\"attributes\":{},\"id\":\"1933\",\"type\":\"LinearScale\"},{\"attributes\":{\"grid_line_color\":{\"value\":null},\"plot\":{\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1936\",\"type\":\"LogTicker\"}},\"id\":\"1939\",\"type\":\"Grid\"},{\"attributes\":{\"source\":{\"id\":\"1958\",\"type\":\"ColumnDataSource\"}},\"id\":\"1963\",\"type\":\"CDSView\"},{\"attributes\":{},\"id\":\"1945\",\"type\":\"PanTool\"},{\"attributes\":{\"background_fill_alpha\":{\"value\":0},\"below\":[{\"id\":\"1935\",\"type\":\"LogAxis\"}],\"border_fill_alpha\":{\"value\":0},\"left\":[{\"id\":\"1940\",\"type\":\"LinearAxis\"}],\"renderers\":[{\"id\":\"1935\",\"type\":\"LogAxis\"},{\"id\":\"1939\",\"type\":\"Grid\"},{\"id\":\"1940\",\"type\":\"LinearAxis\"},{\"id\":\"1944\",\"type\":\"Grid\"},{\"id\":\"1952\",\"type\":\"BoxAnnotation\"},{\"id\":\"1971\",\"type\":\"Legend\"},{\"id\":\"1962\",\"type\":\"GlyphRenderer\"},{\"id\":\"1977\",\"type\":\"GlyphRenderer\"}],\"sizing_mode\":\"scale_width\",\"title\":{\"id\":\"1965\",\"type\":\"Title\"},\"toolbar\":{\"id\":\"1950\",\"type\":\"Toolbar\"},\"toolbar_location\":\"above\",\"x_range\":{\"id\":\"1927\",\"type\":\"DataRange1d\"},\"x_scale\":{\"id\":\"1931\",\"type\":\"LogScale\"},\"y_range\":{\"id\":\"1929\",\"type\":\"DataRange1d\"},\"y_scale\":{\"id\":\"1933\",\"type\":\"LinearScale\"}},\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{\"callback\":null},\"id\":\"1927\",\"type\":\"DataRange1d\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"1975\",\"type\":\"Circle\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"1961\",\"type\":\"Circle\"},{\"attributes\":{\"source\":{\"id\":\"1973\",\"type\":\"ColumnDataSource\"}},\"id\":\"1978\",\"type\":\"CDSView\"},{\"attributes\":{\"callback\":null,\"data\":{\"accuracy\":{\"__ndarray__\":\"tDgU74S65z8=\",\"dtype\":\"float64\",\"shape\":[1]},\"batch_size\":[64],\"color\":[\"#1f77b4\"],\"device\":[\"Tesla V100-SXM2-16GB\"],\"device_mem\":[1568],\"index\":[0],\"model\":[\"alpha_pose_resnet101_v1b_coco\"],\"model_prefix\":[\"alpha\"],\"paper\":[0],\"size\":{\"__ndarray__\":\"AAAAAAAAPkA=\",\"dtype\":\"float64\",\"shape\":[1]},\"throughput\":{\"__ndarray__\":\"Rc2INYKQeUA=\",\"dtype\":\"float64\",\"shape\":[1]},\"workload\":[\"Inference\"]},\"selected\":{\"id\":\"1985\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1986\",\"type\":\"UnionRenderers\"}},\"id\":\"1958\",\"type\":\"ColumnDataSource\"},{\"attributes\":{},\"id\":\"1949\",\"type\":\"ResetTool\"},{\"attributes\":{\"num_minor_ticks\":10},\"id\":\"1936\",\"type\":\"LogTicker\"},{\"attributes\":{\"fill_color\":{\"field\":\"color\"},\"line_color\":{\"field\":\"color\"},\"size\":{\"field\":\"size\",\"units\":\"screen\"},\"x\":{\"field\":\"throughput\"},\"y\":{\"field\":\"accuracy\"}},\"id\":\"1960\",\"type\":\"Circle\"},{\"attributes\":{\"axis_label\":\"#samples/sec\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1966\",\"type\":\"LogTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1936\",\"type\":\"LogTicker\"}},\"id\":\"1935\",\"type\":\"LogAxis\"},{\"attributes\":{},\"id\":\"2057\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"ticker\":null},\"id\":\"1966\",\"type\":\"LogTickFormatter\"},{\"attributes\":{},\"id\":\"1968\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"1931\",\"type\":\"LogScale\"},{\"attributes\":{\"dimension\":1,\"plot\":{\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1941\",\"type\":\"BasicTicker\"}},\"id\":\"1944\",\"type\":\"Grid\"},{\"attributes\":{\"plot\":null,\"text\":\"\",\"text_font_size\":{\"value\":\"100%\"}},\"id\":\"1965\",\"type\":\"Title\"},{\"attributes\":{\"data_source\":{\"id\":\"1973\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1975\",\"type\":\"Circle\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1976\",\"type\":\"Circle\"},\"selection_glyph\":null,\"view\":{\"id\":\"1978\",\"type\":\"CDSView\"}},\"id\":\"1977\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"2056\",\"type\":\"Selection\"},{\"attributes\":{\"callback\":null,\"data\":{\"accuracy\":{\"__ndarray__\":\"wMpOKpQu5z+pbvAqP1nnP1P0KWaFLuc/R1EJn5CA5z9x7kq24DflPwGuXOw9uuY/r/9Aombx5j8=\",\"dtype\":\"float64\",\"shape\":[7]},\"batch_size\":[64,64,64,64,64,64,64],\"color\":[\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\",\"#ff7f0e\"],\"device\":[\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\",\"Tesla V100-SXM2-16GB\"],\"device_mem\":[690,722,674,694,578,714,690],\"index\":[0,1,2,3,4,5,6],\"model\":[\"simple_pose_resnet101_v1b\",\"simple_pose_resnet101_v1d\",\"simple_pose_resnet152_v1b\",\"simple_pose_resnet152_v1d\",\"simple_pose_resnet18_v1b\",\"simple_pose_resnet50_v1b\",\"simple_pose_resnet50_v1d\"],\"model_prefix\":[\"simple\",\"simple\",\"simple\",\"simple\",\"simple\",\"simple\",\"simple\"],\"paper\":[0,0,0,0,0,0,0],\"size\":{\"__ndarray__\":\"10QEi6HmM0C3bdu2bVs0QHH1QGU3qzNA1nmncWD1M0Bt27Zt2zYyQDj779h5PjRA10QEi6HmM0A=\",\"dtype\":\"float64\",\"shape\":[7]},\"throughput\":{\"__ndarray__\":\"SMgNkCkVgUAg23L/JHGAQIKVtDkEb3lAz5kaNw1feECxkyoO/7SdQC/9Wi6iLolAKMwoCxPSh0A=\",\"dtype\":\"float64\",\"shape\":[7]},\"workload\":[\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\",\"Inference\"]},\"selected\":{\"id\":\"2056\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"2057\",\"type\":\"UnionRenderers\"}},\"id\":\"1973\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"callback\":null,\"tooltips\":[[\"Model\",\"@model\"],[\"Throughput\",\"@throughput\"],[\"Accuracy\",\"@accuracy\"],[\"Improvement over Reference\",\"@paper_diff_percent\"],[\"Device memory\",\"@device_mem MB\"]]},\"id\":\"1990\",\"type\":\"HoverTool\"},{\"attributes\":{\"label\":{\"field\":\"model_prefix\"},\"renderers\":[{\"id\":\"1962\",\"type\":\"GlyphRenderer\"}]},\"id\":\"1972\",\"type\":\"LegendItem\"},{\"attributes\":{},\"id\":\"1941\",\"type\":\"BasicTicker\"},{\"attributes\":{\"overlay\":{\"id\":\"1952\",\"type\":\"BoxAnnotation\"}},\"id\":\"1947\",\"type\":\"BoxZoomTool\"},{\"attributes\":{\"axis_label\":\"Accuracy\",\"axis_label_text_font_size\":{\"value\":\"1em\"},\"formatter\":{\"id\":\"1968\",\"type\":\"BasicTickFormatter\"},\"major_label_text_font_size\":{\"value\":\"0.5em\"},\"plot\":{\"id\":\"1926\",\"subtype\":\"Figure\",\"type\":\"Plot\"},\"ticker\":{\"id\":\"1941\",\"type\":\"BasicTicker\"}},\"id\":\"1940\",\"type\":\"LinearAxis\"},{\"attributes\":{\"callback\":null},\"id\":\"1929\",\"type\":\"DataRange1d\"}],\"root_ids\":[\"1926\"]},\"title\":\"Bokeh Application\",\"version\":\"1.0.4\"}}\n        </script>\n        <script type=\"text/javascript\">\n          (function() {\n            var fn = function() {\n              Bokeh.safely(function() {\n                (function(root) {\n                  function embed_document(root) {\n                    \n                  var docs_json = document.getElementById('2213').textContent;\n                  var render_items = [{\"docid\":\"1b922361-77d4-4c72-9f1a-331d591c2315\",\"roots\":{\"1926\":\"808c0fe4-add6-40ea-b556-ed10866672e8\"}}];\n                  root.Bokeh.embed.embed_items(docs_json, render_items);\n                \n                  }\n                  if (root.Bokeh !== undefined) {\n                    embed_document(root);\n                  } else {\n                    var attempts = 0;\n                    var timer = setInterval(function(root) {\n                      if (root.Bokeh !== undefined) {\n                        embed_document(root);\n                        clearInterval(timer);\n                      }\n                      attempts++;\n                      if (attempts > 100) {\n                        console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n                        clearInterval(timer);\n                      }\n                    }, 10, root)\n                  }\n                })(window);\n              });\n            };\n            if (document.readyState != \"loading\") fn();\n            else document.addEventListener(\"DOMContentLoaded\", fn);\n          })();\n        </script>\n    \n  </body>\n  \n</html>\n"
 }
]
```

```{.python .input  n=13}
with open(os.path.join(os.path.abspath(''), 'pose_throughputs.html'), 'wt') as f:
    f.write(html)
```

```{.json .output n=13}
[
 {
  "ename": "NameError",
  "evalue": "name 'os' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-13-a108af63beed>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mwith\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mabspath\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m''\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'pose_throughputs.html'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'wt'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m     \u001b[0mf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhtml\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'os' is not defined"
  ]
 }
]
```
