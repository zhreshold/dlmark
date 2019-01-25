# GluonCV: detection

We benchmark the convolutional neural networks provided by the [GluonCV modelzoo](https://gluon-cv.mxnet.io/model_zoo/index.html#) for object detection.

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=1}
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

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "(array(['ssd_512_mobilenet1.0_coco', 'ssd_300_vgg16_atrous_coco',\n        'ssd_512_vgg16_atrous_coco', 'ssd_512_resnet50_v1_coco',\n        'yolo3_darknet53_coco@320', 'yolo3_darknet53_coco@416',\n        'yolo3_darknet53_coco@608', 'faster_rcnn_resnet50_v1b_coco',\n        'faster_rcnn_resnet101_v1d_coco'], dtype=object),\n array(['Tesla V100-SXM2-16GB'], dtype=object))"
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

data = thr[(thr.device==devices[0]) & (thr.batch_size.isin([1,2,4,8,16,32,64,128]))]
# show(plot.batch_size_vs_throughput_grid(data, models[::1]))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/html": "\n    <div class=\"bk-root\">\n        <a href=\"https://bokeh.pydata.org\" target=\"_blank\" class=\"bk-logo bk-logo-small bk-logo-notebook\"></a>\n        <span id=\"bae91df9-0497-4439-b681-75722f399e32\">Loading BokehJS ...</span>\n    </div>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "application/javascript": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof (root._bokeh_onload_callbacks) === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  var JS_MIME_TYPE = 'application/javascript';\n  var HTML_MIME_TYPE = 'text/html';\n  var EXEC_MIME_TYPE = 'application/vnd.bokehjs_exec.v0+json';\n  var CLASS_NAME = 'output_bokeh rendered_html';\n\n  /**\n   * Render data to the DOM node\n   */\n  function render(props, node) {\n    var script = document.createElement(\"script\");\n    node.appendChild(script);\n  }\n\n  /**\n   * Handle when an output is cleared or removed\n   */\n  function handleClearOutput(event, handle) {\n    var cell = handle.cell;\n\n    var id = cell.output_area._bokeh_element_id;\n    var server_id = cell.output_area._bokeh_server_id;\n    // Clean up Bokeh references\n    if (id !== undefined) {\n      Bokeh.index[id].model.document.clear();\n      delete Bokeh.index[id];\n    }\n\n    if (server_id !== undefined) {\n      // Clean up Bokeh references\n      var cmd = \"from bokeh.io.state import curstate; print(curstate().uuid_to_server['\" + server_id + \"'].get_sessions()[0].document.roots[0]._id)\";\n      cell.notebook.kernel.execute(cmd, {\n        iopub: {\n          output: function(msg) {\n            var element_id = msg.content.text.trim();\n            Bokeh.index[element_id].model.document.clear();\n            delete Bokeh.index[element_id];\n          }\n        }\n      });\n      // Destroy server and session\n      var cmd = \"import bokeh.io.notebook as ion; ion.destroy_server('\" + server_id + \"')\";\n      cell.notebook.kernel.execute(cmd);\n    }\n  }\n\n  /**\n   * Handle when a new output is added\n   */\n  function handleAddOutput(event, handle) {\n    var output_area = handle.output_area;\n    var output = handle.output;\n\n    // limit handleAddOutput to display_data with EXEC_MIME_TYPE content only\n    if ((output.output_type != \"display_data\") || (!output.data.hasOwnProperty(EXEC_MIME_TYPE))) {\n      return\n    }\n\n    var toinsert = output_area.element.find(`.${CLASS_NAME.split(' ')[0]}`);\n\n    if (output.metadata[EXEC_MIME_TYPE][\"id\"] !== undefined) {\n      toinsert[0].firstChild.textContent = output.data[JS_MIME_TYPE];\n      // store reference to embed id on output_area\n      output_area._bokeh_element_id = output.metadata[EXEC_MIME_TYPE][\"id\"];\n    }\n    if (output.metadata[EXEC_MIME_TYPE][\"server_id\"] !== undefined) {\n      var bk_div = document.createElement(\"div\");\n      bk_div.innerHTML = output.data[HTML_MIME_TYPE];\n      var script_attrs = bk_div.children[0].attributes;\n      for (var i = 0; i < script_attrs.length; i++) {\n        toinsert[0].firstChild.setAttribute(script_attrs[i].name, script_attrs[i].value);\n      }\n      // store reference to server id on output_area\n      output_area._bokeh_server_id = output.metadata[EXEC_MIME_TYPE][\"server_id\"];\n    }\n  }\n\n  function register_renderer(events, OutputArea) {\n\n    function append_mime(data, metadata, element) {\n      // create a DOM node to render to\n      var toinsert = this.create_output_subarea(\n        metadata,\n        CLASS_NAME,\n        EXEC_MIME_TYPE\n      );\n      this.keyboard_manager.register_events(toinsert);\n      // Render to node\n      var props = {data: data, metadata: metadata[EXEC_MIME_TYPE]};\n      render(props, toinsert[0]);\n      element.append(toinsert);\n      return toinsert\n    }\n\n    /* Handle when an output is cleared or removed */\n    events.on('clear_output.CodeCell', handleClearOutput);\n    events.on('delete.Cell', handleClearOutput);\n\n    /* Handle when a new output is added */\n    events.on('output_added.OutputArea', handleAddOutput);\n\n    /**\n     * Register the mime type and append_mime function with output_area\n     */\n    OutputArea.prototype.register_mime_type(EXEC_MIME_TYPE, append_mime, {\n      /* Is output safe? */\n      safe: true,\n      /* Index of renderer in `output_area.display_order` */\n      index: 0\n    });\n  }\n\n  // register the mime type if in Jupyter Notebook environment and previously unregistered\n  if (root.Jupyter !== undefined) {\n    var events = require('base/js/events');\n    var OutputArea = require('notebook/js/outputarea').OutputArea;\n\n    if (OutputArea.prototype.mime_types().indexOf(EXEC_MIME_TYPE) == -1) {\n      register_renderer(events, OutputArea);\n    }\n  }\n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) { callback() });\n    }\n    finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.info(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(js_urls, callback) {\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.log(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.log(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = js_urls.length;\n    for (var i = 0; i < js_urls.length; i++) {\n      var url = js_urls[i];\n      var s = document.createElement('script');\n      s.src = url;\n      s.async = false;\n      s.onreadystatechange = s.onload = function() {\n        root._bokeh_is_loading--;\n        if (root._bokeh_is_loading === 0) {\n          console.log(\"Bokeh: all BokehJS libraries loaded\");\n          run_callbacks()\n        }\n      };\n      s.onerror = function() {\n        console.warn(\"failed to load library \" + url);\n      };\n      console.log(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.getElementsByTagName(\"head\")[0].appendChild(s);\n    }\n  };var element = document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\");\n  if (element == null) {\n    console.log(\"Bokeh: ERROR: autoload.js configured with elementid 'bae91df9-0497-4439-b681-75722f399e32' but no matching script tag was found. \")\n    return false;\n  }\n\n  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-0.12.10.min.js\"];\n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    \n    function(Bokeh) {\n      \n    },\n    function(Bokeh) {\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.css\");\n    }\n  ];\n\n  function run_inline_js() {\n    \n    if ((root.Bokeh !== undefined) || (force === true)) {\n      for (var i = 0; i < inline_js.length; i++) {\n        inline_js[i].call(root, root.Bokeh);\n      }if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.log(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(js_urls, function() {\n      console.log(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));",
   "application/vnd.bokehjs_load.v0+json": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof (root._bokeh_onload_callbacks) === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  \n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) { callback() });\n    }\n    finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.info(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(js_urls, callback) {\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.log(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.log(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = js_urls.length;\n    for (var i = 0; i < js_urls.length; i++) {\n      var url = js_urls[i];\n      var s = document.createElement('script');\n      s.src = url;\n      s.async = false;\n      s.onreadystatechange = s.onload = function() {\n        root._bokeh_is_loading--;\n        if (root._bokeh_is_loading === 0) {\n          console.log(\"Bokeh: all BokehJS libraries loaded\");\n          run_callbacks()\n        }\n      };\n      s.onerror = function() {\n        console.warn(\"failed to load library \" + url);\n      };\n      console.log(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.getElementsByTagName(\"head\")[0].appendChild(s);\n    }\n  };var element = document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\");\n  if (element == null) {\n    console.log(\"Bokeh: ERROR: autoload.js configured with elementid 'bae91df9-0497-4439-b681-75722f399e32' but no matching script tag was found. \")\n    return false;\n  }\n\n  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-0.12.10.min.js\"];\n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    \n    function(Bokeh) {\n      \n    },\n    function(Bokeh) {\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-0.12.10.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.10.min.css\");\n      console.log(\"Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.css\");\n      Bokeh.embed.inject_css(\"https://cdn.pydata.org/bokeh/release/bokeh-tables-0.12.10.min.css\");\n    }\n  ];\n\n  function run_inline_js() {\n    \n    if ((root.Bokeh !== undefined) || (force === true)) {\n      for (var i = 0; i < inline_js.length; i++) {\n        inline_js[i].call(root, root.Bokeh);\n      }if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"bae91df9-0497-4439-b681-75722f399e32\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.log(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(js_urls, function() {\n      console.log(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

The throughput increases with the batch size in log scale. The device memory, as exepcted, also increases linearly with the batch size. But note that, due to the pooled memory mechanism in MXNet, the measured device memory usage might be different to the actual memory usdage.

One way to measure the actual device memory usage is finding the largest batch size we can run.

```{.python .input  n=3}
bs = pd.concat([dm.benchmark.load_results(prefix + '.py__benchmark_max_batch_size.json') for prefix in prefixs])
# show(plot.max_batch_size(bs))
```

## Throughput on various hardware

```{.python .input  n=4}
#show(plot.throughput_vs_device(thr[(thr.model=='AlexNet')]))
```

```{.python .input  n=5}
#show(plot.throughput_vs_device(thr[(thr.model=='ResNet-v2-50')]))
```

```{.python .input  n=6}
paper_results = {
    'faster_rcnn_resnet50_v1b_coco': 36.5,
    'yolo3_darknet53_coco@608': 33.0,
    'yolo3_darknet53_coco@416': 31.0,
    'yolo3_darknet53_coco@320': 28.6,
}
```

### Prediction mAP versus throughput

We measture the prediction mean AP of each model using the MSCOCO 2017 validation dataset. Then plot the results together with the best throughput across various batch-sizes(if applicable). We colorize models from the same family with the same color.

```{.python .input  n=7}
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
```

```{.python .input  n=8}
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
```

```{.python .input  n=9}
maps = pd.concat([dm.benchmark.load_results(prefix + '*map.json') for prefix in prefixs])
thr = thr.reset_index(drop=True)
thr_sorted = thr.sort_values(by='throughput', ascending=False).drop_duplicates(['model'])
thr_sorted['best_throughput_batch_size'] = thr.batch_size[thr_sorted.index]

data = thr_sorted[(thr_sorted.model.isin(maps.model)) &
           (thr_sorted.device.isin(maps.device))]
data1 = data.set_index('model').join(maps[['model','map', 'map_per_class']].set_index('model'))
data = data.set_index('model').join(maps[['model','map']].set_index('model'))
data['model_prefix'] = [i[:i.find('_')] if i.find('_') > 0 and not i.startswith('faster') else 'faster_rcnn' for i in data.index]
# print(data['model_prefix'])
# print(data['model_prefix']['faster_rcnn_resnet50_v1b_coco'])
# print(data['model_prefix'])

# pp = throughput_vs_map(data)
src = make_dataset(data, data.index.values.tolist())
pp = make_plot(src)
pp.sizing_mode = 'scale_width'
show(pp)


# from bokeh.io import export_png
# pp.background_fill_color = "beige"
# pp.background_fill_alpha = 0.5
# export_png(pp, filename="detection.png")
```

```{.json .output n=9}
[
 {
  "ename": "NameError",
  "evalue": "name 'dict' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-9-bc0b493626b4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;31m# pp = throughput_vs_map(data)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0;32mdel\u001b[0m \u001b[0mdict\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     17\u001b[0m \u001b[0msrc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmake_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtolist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0mpp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmake_plot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'dict' is not defined"
  ]
 }
]
```

```{.python .input  n=10}
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
    
    colors = Category20[max(len(data.index),3)]

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
    
    
# p2 = plot_bar(data1)
# show(p2)
```

```{.python .input  n=11}
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

html = file_html(pp, CDN, "my plot")
print(html)
```

```{.python .input  n=12}
# per class perf comparison

```
