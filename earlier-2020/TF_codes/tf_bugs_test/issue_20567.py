"""
tf.py_func(my_py_func, args, dype) returns a tensor of known type but unknown shape.
When my_py_func runs, py_func will check that the type is what's expected and fail otherwise.

But, if you set an incorrect shape with .set_shape there's no warning. Having a loud warning would have saved me a few hours yesterday.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def my_py_func(x):
    return x


def main(argv):
    del argv

    print(tf.GIT_VERSION, tf.VERSION)

    t = tf.constant(0, dtype=tf.int64)
    print('tf.constant(0):', t)
    t = tf.py_func(my_py_func, [t], tf.int64)
    print('tf.py_func:', t)
    t.set_shape([100, 100])
    print('t.set_shape:', t)

    with tf.Session() as sess:
        t = sess.run(t)
        print('sess.run(t)', t)


def main2():
    x = tf.placeholder(tf.float32)  # Unknown shape
    y = tf.identity(x)
    y.set_shape([100, 100])
    print(x)
    print(y)
    with tf.Session() as sess:
        print(sess.run(y, feed_dict={x: 10}))

def main3():
    x = tf.constant([1, 1, 2])
    y, idx = tf.unique(x)
    y.set_shape([100])
    print(x)
    print(y)
    with tf.Session() as sess:
        print(sess.run(y))

if __name__ == '__main__':
    main3()
