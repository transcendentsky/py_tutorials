#coding:utf-8
"""
Some explaination of tf.identity
"""
import tensorflow as tf

def main1():
    x = tf.Variable(1.0)
    # y = tf.Variable(0.0)

    # 返回一个op，表示给变量x加1的操作
    x_plus_1 = tf.assign_add(x, 1)

    # control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前，
    # 先执行control_dependencies参数中的内容（在这里就是 x_plus_1），这里的解释不准确，先接着看。。。
    with tf.control_dependencies([x_plus_1]):
        y = x
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        for i in xrange(5):
            print(y.eval())
            # 相当于sess.run(y)，按照我们的预期，由于control_dependencies的作用，
            # 所以应该执行print前都会先执行x_plus_1，但是这种情况会出问题


def main2():
    x = tf.Variable(1.0)
    # y = tf.Variable(0.0)
    x_plus_1 = tf.assign_add(x, 1)

    with tf.control_dependencies([x_plus_1]):
        y = tf.identity(x)  # 修改部分
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        for i in xrange(5):
            print(y.eval())

main2()