# cifar10 / cnn

import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)    # (50000, 32, 32, 3)
# print(x_test.shape)     # (10000, 32, 32, 3)
# print(y_train.shape)    # (50000, 1)
# print(y_test.shape)     # (10000, 1)

x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32') / 255.0

# y원핫

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)  # (50000, 10)
# print(y_test.shape)   # (10000, 10)


learning_rate = 0.2
training_epochs = 10
batch_size = 100
total_batch = int(len(x_train) / batch_size)

x = tf.compat.v1.placeholder('float32', [None, 32,32,3])
y = tf.compat.v1.placeholder('float32', [None, 10])

#2. 모델구성
w = tf.compat.v1.get_variable("w1", shape=[2,2,3,32])
layer = tf.nn.conv2d(x, w, strides = [1,2,2,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool2d(layer, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

w = tf.compat.v1.get_variable("w2", shape = [2,2,32,64])
layer = tf.nn.conv2d(layer, w, strides = [1,2,2,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool2d(layer, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

layer = tf.reshape(layer, [-1,layer.shape[1]*layer.shape[2]*layer.shape[3]])

w = tf.compat.v1.get_variable("w4", shape = [layer.shape[1], 32])
layer = tf.nn.selu(tf.matmul(layer, w))

w = tf.compat.v1.get_variable("w5", shape = [32,10])
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis = 1))
opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs): # 20
        avg_cost = 0

        for i in range(total_batch):    # 500
            start = i * batch_size
            end = start + batch_size
            
            batch_xs, batch_ys = x_train[start:end], y_train[start:end]

            feed_dict = {x:batch_xs, y:batch_ys}
            c, _ = sess.run([cost, opt], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print(f"epoch: {epoch+1}\t cost: {avg_cost}")

    print("훈련 끝!")

    pred = tf.equal(tf.arg_max(h,1), tf.argmax(y,1))

    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))