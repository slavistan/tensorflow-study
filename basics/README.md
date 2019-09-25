```sos
# This notebook was tested with tf 1.11.0
from importlib_metadata import version
assert(version("tensorflow") == "1.11.0")
```

# Generate a graph, run and visualize it


```sos
# By convention tensorflow is always aliased as `tf`
import tensorflow as tf


# Construction phase: Build the graph by defining nodes.
x1 = tf.Variable(3, name="x1") # Define a leaf node
x2 = tf.Variable(4, name="x2") # The name is displayed during visualization
y = x1+x2
z = x1*y


# Computation phase: Run the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run() # Variable initialization
    print("y =", y.eval()) # Isolated evaluation
    print("[y, z] =", sess.run([y, z])) # Optimized, combined evaluation
    tf.summary.FileWriter("/tmp/tf/graph", sess.graph) # Export graph information

    
# Executing the above code multiple times in a notebook will add distinct
# nodes to the graph. Thus we reset the graph post-execution.
tf.reset_default_graph()
```

#### Initializing leaf nodes

All nodes can be initialized collectively via a call to `tf.global_variables_initializer().run()` or, individually, using `sess.run(x1.initializer)`.

#### Evaluating nodes

Nodes are evaluated via `sess.run(y)` or, equivalently, `y.run()`. Each evaluation causes al dependent nodes to be evaluated from scratch - intermediate results are not cached. In order to reuse results from dependent nodes use `sess.run([y, z])`.

#### Visualizing a graph

Graph information is exported for visualization using `tf.summary.FileWriter("/path/to/graph", sess.graph)`. Afterwards a tensorboard session can be invoked from a shell using `tensorboard --logdir="/path/to/graph"` and opening the displayed URL in a browser. 

# Save and inspect models

## Simple save and load


```sos
import tensorflow as tf

# Define a simple graph
tf.reset_default_graph()
x = tf.Variable(3.14, name="x")
y = 2 * x

# Save the graph using the `simple_save` wrapper
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.saved_model.simple_save(sess,
                "./simple_save_model",
                inputs={"x": x},
                outputs={"y": y})

```

## Save using a `builder`


```sos
import tensorflow as tf
import os
import shutil


# Reset everything so this cell can be re-run arbitrarily
tf.reset_default_graph()# Reset tf graph
export_dir = "./builder_model" # delete the export directory
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

    
# Define a simple graph which doubles the input value. Use
# placeholders for inputs, as it should be done. Variables are
# used to represent trained parameters.
tf.reset_default_graph()
x1 = tf.placeholder(name="input_x1", dtype="float", shape=())
x2 = tf.placeholder(name="input_x2", dtype="float", shape=())
y = tf.add(x1, x2, name="output_y")


# A signature defines a model's inputs and outputs. The signature's names will
# be used when passing input variables (not the names given to the tensors
# during model construction).
my_default_sig=tf.saved_model.signature_def_utils.predict_signature_def({"sig_input_x1": x1, "sig_input_x2": x2}, {"sig_output_y": y})


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir) # A `builder` class is used to save models.
    builder.add_meta_graph_and_variables(sess,
                                  ["footag"], # tags
                                  signature_def_map={"my_funny_sig": my_default_sig})
builder.save() # actually save the model
```

    INFO:tensorflow:No assets to save.
    INFO:tensorflow:No assets to write.
    INFO:tensorflow:SavedModel written to: ./builder_model/saved_model.pb





    b'./builder_model/saved_model.pb'



## Inspect a `SavedModel` via the **saved_model_cli**


```sos
# List information about information
saved_model_cli show --dir builder_model --all
```


```sos
# Perform a computation. Our model adds the two inputs.
x1="3.00";x2="2.00"
saved_model_cli run --dir "builder_model" \
    --tag_set "footag" \
    --signature_def "my_funny_sig" \
    --input_exprs="sig_input_x1=$x1;sig_input_x2=$x2"
```

    2019-09-25 18:38:15.892995: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    Result for output key sig_output_y:
    5.0



```sos
saved_model_cli run --help
```

# Load a `SavedModel`


```sos
%reset -f
import tensorflow as tf
tf.reset_default_graph()


# Load the model and list the tensors' name we brought into scope. There
# aren't any tf.Variables which need initialization, hence no global init.
with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["footag"], "./builder_model")
    # We address tensors by name after loading a model
    print(sess.run('output_y:0', feed_dict = {"input_x1:0": 3.14, "input_x2:0": 4.4}))
    print(tf.get_default_graph().get_tensor_by_name("output_y:0").eval(feed_dict={"input_x1:0": 1.5, "input_x2:0": 3}))
```

    INFO:tensorflow:Saver not created because there are no variables in the graph to restore
    INFO:tensorflow:The specified SavedModel has no variables; no checkpoints were restored.
    7.54
    4.5


# Serve a model


```sos

```
