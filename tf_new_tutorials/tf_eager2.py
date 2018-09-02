#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
tfe.enable_eager_execution()

from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard

# from IPython import embed
# from tensorflow.data.Datasets

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, kernel_initializer='he_normal'))
    return model


def cross_entropy(model_output, label_batch):
    loss = tf.reduce_mean(
        -tf.reduce_sum(label_batch * tf.log(model_output),
                       reduction_indices=[1]))
    return loss


if __name__ == '__main__':
    data = input_data.read_data_sets("/media/trans/mnt/data/MNIST/", one_hot=True)
    print type(data)
    print type(data.train.images)
    # exit(0)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # In Docs , should be tensors generally, but numpy type works.
    train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels)) \
        .map(lambda x, y: (x, tf.cast(y, tf.float32))) \
        .shuffle(buffer_size=1000) \
        .batch(100)

    model = build_model()

    optimizer = tf.train.GradientDescentOptimizer(0.5)

    for step, (image_batch, label_batch) in enumerate(tfe.Iterator(train_ds)):
        exit(0)
        with tf.GradientTape() as tape:
            image_batch = tf.reshape(image_batch, (100,28,28,1))
            output = model(image_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_batch))
        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables))
        print("step: {}  loss: {}".format(step, loss.numpy()))

    # model_test_output = model(data.test.images)
    # model_test_label = data.test.labels
    # correct_prediction = tf.equal(tf.argmax(model_test_output, 1), tf.argmax(model_test_label, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("test accuracy = {}".format(accuracy.numpy()))
