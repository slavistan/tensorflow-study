
### Generate a graph, run and visualize it


```python
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

    y = 7
    [y, z] = [7, 21]


#### Initializing leaf nodes

All nodes can be initialized collectively via a call to `tf.global_variables_initializer().run()` or, individually, using `sess.run(x1.initializer)`.

#### Evaluating nodes

Nodes are evaluated via `sess.run(y)` or, equivalently, `y.run()`. Each evaluation causes al dependent nodes to be evaluated from scratch - intermediate results are not cached. In order to reuse results from dependent nodes use `sess.run([y, z])`.

#### Visualizing a graph

Graph information is exported for visualization using `tf.summary.FileWriter("/path/to/graph", sess.graph)`. Afterwards a tensorboard session can be invoked from a shell using `tensorboard --logdir="/path/to/graph"`.
