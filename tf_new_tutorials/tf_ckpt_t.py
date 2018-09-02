import tensorflow as tf
import numpy as np

####  Illustrated Variable  ####
with tf.variable_scope("demo",reuse=tf.AUTO_REUSE):
    demo_weights = tf.get_variable(name='weights', shape=[128, 64])
# initializer = tf.truncated_normal_initializer(
# stddev = 1.0 / np.sqrt(float(128)))

print("demo_weights")
print(demo_weights)
# print(demo_weights.name)
# print(demo_weights.initial_value)
# print(demo_weights.dtype)
# print(demo_weights.shape)
# print(demo_weights.initializer)

with tf.variable_scope("demo",reuse=tf.AUTO_REUSE):
    demo_weights2 = tf.get_variable(name='weights', shape=[128, 64])
print(demo_weights2)

a,b,c,d=tf.nn.top_k([1], 1)
print a # => 'TopKV2:0'
print d # => 'TopKV2:1'

# print(demo_weights.op)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(demo_weights))