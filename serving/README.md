# Start a model server and get a prediction

https://www.tensorflow.org/tfx/serving/api_rest

# Save a model


```sos
import tensorflow as tf
import os

# Running a model server requires an explicit model version, even if only a
# single version is available. The version is encoded as subdirectories within
# the model directory, such as `model/1/...` and `model/2/...`.
#
# This code increments the version each time it's executed.
model_root = "./model/"
if not os.path.exists(model_root):
    os.mkdir("./model/")
versions = [int(x) for x in [dirs for (r, dirs, f) in os.walk(model_root) if r==model_root][0]]
if len(versions) == 0:
    export_dir = model_root + "1/"
else:
    export_dir = model_root + str(max(versions) + 1) + "/"
print("\033[32;1m" + "Export dir: " + "\033[0m" + export_dir)

# Reset everything so this cell can be re-run arbitrarily
tf.reset_default_graph()# Reset tf graph

    
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
sig=tf.saved_model.signature_def_utils.predict_signature_def({"sig_input_x1": x1, "sig_input_x2": x2}, {"sig_output_y": y})
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir) # A `builder` class is used to save models.
    builder.add_meta_graph_and_variables(sess,
                                  ["serve"], # tags
                                  signature_def_map={"serving_default": sig})
builder.save() # actually save the model
```

## Start the model server

Start the model server using `nohup` when inside the notebook.


```sos
MODEL_DIR="$(realpath ./model)"
MODEL_NAME="addition"
# TODO: Make nohup not create a nohup.log
nohup tensorflow_model_server \
  --rest_api_port=8501        \
  --model_name=addition       \
  --model_base_path="${MODEL_DIR}" > /dev/null 2>&1 &
model_server_url="localhost:8501/v1/models/${MODEL_NAME}"
```

## Probe the model server for its metadata


```sos
curl --url "${model_server_url}/metadata"
```

## Get a prediction


```sos
curl --url "${model_server_url}:predict" \
    --request "POST" \
    --data '{"instances": [{"sig_input_x1":[1, 2],"sig_input_x2":[1, 4]}]}'
```
