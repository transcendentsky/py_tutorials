#coding:utf-8
import tensorflow as tf
print(tf.__version__)
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import numpy as np

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(Flatten())
    model.add(Dense(2, kernel_initializer='he_normal'))
    return model

def criterion(x, y):
    loss = tf.reduce_sum(tf.losses.huber_loss(
        x, y, delta=1.0))
    return loss

class LocDataLoader(object):
    def __init__(self, num=10000, size=10, batch_size=32):
        self.size = 10
        self.batch_size = batch_size
        self.yield_num = num

    def get_item(self):
        input = np.ones((self.size,self.size))
        loc = np.random.randint(0,self.size, (2,))
        input[loc[0], loc[1]] = 9
        return input, loc

    def forward(self):
        inputs = []
        locs = []
        for i in range(self.batch_size):
            input, loc = self.get_item()
            inputs.append(input)
            locs.append(loc)
        inputs = np.array(inputs)
        locs = np.array(locs)
        return inputs, locs

    def yield_data(self):
        for i in range(self.yield_num):
            yield self.forward()

if __name__ == '__main__':
    model = build_model()
    loader = LocDataLoader()
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    step = 0
    for (image_batch, label_batch) in iter(loader.yield_data()):
        with tf.GradientTape() as tape:
            image_batch = tf.reshape(image_batch, (100,28,28,1))
            output = model(image_batch)
            loss = criterion(output, label_batch)
        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables))
        print("step: {}  loss: {}".format(step, loss.numpy()))
        step += 1