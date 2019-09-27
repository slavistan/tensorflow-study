**Developed using**

- Local [tensorflow_model_server][1] binary version 1.14 
- Python 3.7.3
  - tensorflow==1.14.0
  - tensorflow-serving-api==1.14.0
  - grpcio==1.24.0

[1]: https://www.tensorflow.org/tfx/serving/setup

# Create a model - Launch an unsecured server - Get predictions using the REST and RPC APIs

https://www.tensorflow.org/tfx/serving/api_rest

## Create a model

We create a simple model which implements an addition $(x_1, x_2) \mapsto x_1 + x_2$


```sos
%reset -f
import tensorflow as tf
import os

# Running a model server requires an explicit model version, even if only a
# single version is available. The version is encoded as subdirectories within
# the model directory, such as `model/1/...` and `model/2/...`.
# This code increments the version each time it's executed.
model_root = "./model/"
if not os.path.exists(model_root):
    os.mkdir("./model/")
versions = [int(x) for x in [dirs for (r, dirs, f) in os.walk(model_root) if r==model_root][0]]
if len(versions) == 0:
    export_dir = model_root + "1/"
else:
    export_dir = model_root + str(max(versions) + 1) + "/"


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
builder.save()
```

## Start the model server

Start the model server using `nohup` when inside the notebook.


```sos
model_dir="$(realpath ./model)" # we need the full path
model_name="addition"
nohup tensorflow_model_server \
  --port=8500                 \
  --rest_api_port=8501        \
  --model_name=addition       \
  --model_base_path="${model_dir}" > /dev/null 2>&1 &
```

## Probe the model server for its metadata


```sos
model_server_url="localhost:8501/v1/models/${model_name}"
curl --url "${model_server_url}/metadata"
```

## Get a prediction using the REST api


```sos
curl -s --url "${model_server_url}:predict" \
    --request "POST" \
    --data '{"instances": [{"sig_input_x1":[1, 2, 7],"sig_input_x2":[1, -4, 3.14]}]}'
```

## Get prediction using the gRPC api


```sos
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

# Configure the request, i.e. set up
# - the model name
# - the signature (which set of in- and outputs to choose)
# - the input data
request = predict_pb2.PredictRequest()
request.model_spec.name = 'addition'
request.model_spec.signature_name = 'serving_default'
request.inputs['sig_input_x1'].CopyFrom(
    tf.contrib.util.make_tensor_proto([1, 2], dtype="float"))
request.inputs['sig_input_x2'].CopyFrom(
    tf.contrib.util.make_tensor_proto([1, 2], dtype="float"))

# Create the stub.
# "A stub in distributed computing is a piece of code that 
#  converts parameters passed between client and server during
#  a remote procedure call.""
channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Shoot. Pass an optional RPC timout.
result=stub.Predict(request, 10)  # TODO: What's the timeout's unit? Seconds?
print(result)
```


```sos
# Select individual values
print(result.outputs['sig_output_y'].float_val)
```
