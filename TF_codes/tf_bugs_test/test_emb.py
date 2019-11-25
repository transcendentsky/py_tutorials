#coding:utf-8
import tensorflow as tf
import numpy as np

### embedding_lookup 是找到对应的元素，if the index is overflow, return [0 , ...]
c = np.arange(10*16).reshape(10,16)
b = tf.nn.embedding_lookup(c, [1,2,3,10,11])

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print sess.run(b)
  print c