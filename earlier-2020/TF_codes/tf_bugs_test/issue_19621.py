"""
System information

    Have I written custom code (as opposed to using a stock example script provided in TensorFlow): No
    OS Platform and Distribution (e.g., Linux Ubuntu 16.04): Windows
    TensorFlow installed from (source or binary): binary
    TensorFlow version (use command below): v1.8.0-0-g93bc2e2072
    Python version: 3.6.3
    Bazel version (if compiling from source):
    GCC/Compiler version (if compiling from source):
    CUDA/cuDNN version: 9.0/7.0
    GPU model and memory: 4G/Quadro M1000M
    Exact command to reproduce: -

Describe the problem

The graph can be executed with AdamOptimizer, thus the graph input is invalid.
In the example an out-of-bound embedding index is passed. Other tested optimizer
(RMSP,Ada) yield: InvalidArgumentError: indices[0,0] = 10 is not in [0, 10)
"""

import tensorflow as tf
import numpy as np

num_factors = 16
num_embed = 10

# size=10, dim=16
def embed_helper(inputs, size, dim, name=None):
  std = np.square(2. / dim)
  print(std)
  test_emb = tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)
  return tf.nn.embedding_lookup(test_emb, inputs), test_emb


graph = tf.Graph()
with graph.as_default():
  x = tf.placeholder(tf.int32, shape=(None, 1))
  x_emb, test_emb = embed_helper(x, num_embed, num_factors, name=None)
  very_complicated_net = tf.square(tf.subtract(x_emb, tf.constant(3.)))

  # RMSP and the other tested optimizer yield InvalidArgumentError - AdamOptimizer does not
  # -> InvalidArgumentError (see above for traceback): indices[0,0] = 10 is not in [0, 10)
  opt = tf.train.RMSPropOptimizer(learning_rate=0.01)
  # opt = tf.train.AdamOptimizer(learning_rate=0.01)

  graph_step = opt.minimize(very_complicated_net)
  init = tf.global_variables_initializer()

session = tf.Session(config=None, graph=graph)
session.run(init)

# 10 is not in [0,1,..,9]
index_batch_out_of_bound = np.array([10]).reshape(-1, 1)
# index_batch_out_of_bound = np.arange(10).reshape(-1, 1)

x_feed_dict = {
  x: index_batch_out_of_bound
}

# r = session.run(graph_step, x_feed_dict)
# print r  # session.run(graph_step, x_feed_dict)

r = session.run(x, x_feed_dict)
print r
r = session.run(x_emb, x_feed_dict)
print r
r = session.run(very_complicated_net, x_feed_dict)
print r
r = session.run(test_emb, x_feed_dict)
print r