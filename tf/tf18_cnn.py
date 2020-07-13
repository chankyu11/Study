import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# print(x_train.shape)#(60000, 28, 28)
# print(y_train.shape)#(60000,)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(-1, 28,28,1) / 255.0
x_test = x_test.reshape(-1,28,28,1) / 255.0

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)

x = tf.placeholder(tf.float32, [None, 28,28,1])

x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)   # dropout

# 1. w1 = tf.Variable(tf.random_normal([784, 512], name = 'bias'))
# 2. w1 = tf.get_variable("w1", shape=[784, 512])
# 1과 2는 같은 뜻

# layer 1
w1 = tf.get_variable("w1", shape=[3,3,1,32])
# keras = Conv2D(32, (3,3), input = (28, 28,1))
# [3,3,1,32] 3, 3은 커널사이즈, 1은 채널, 32는 아웃풋
L1 = tf.nn.conv2d(x_img, w1, strides = [1,1,1,1], padding = 'SAME')
# b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# L1 = tf.nn.dropout(L1, keep_prob = keep_prob)


# # layer 2
w2 = tf.get_variable("w2", shape=[3,3,32,64])
# 3, 3, 32, 64 여기서 32는 받아온거 채널은 상위 레이어의 아웃풋

L2 = tf.nn.conv2d(L1, w2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# Flatten
L2_flat = tf.reshape(L2, [-1, 7*7*64])

w3 = tf.get_variable("w3", shape=[7*7*64, 10])
                    #  initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L2_flat, w3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size                   # 0
        end = start + batch_size    # 100

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        feed_dict = {x: batch_xs, y:batch_ys}   # , keep_prob: 0.7
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    
    print('epoch: ', '%04d'% (epoch + 1),
          'cost: ', '{:.9f}'.format(avg_cost))
print("끝!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("ACC:", sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))