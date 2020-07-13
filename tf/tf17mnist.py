
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
# print(x_train.shape)#(60000, 28, 28)
# print(y_train.shape)#(60000,)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_test.shape)
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]) / 255.0
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]) / 255.0

learning_rate = 0.0000001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)
keep_prob = tf.placeholder('float32')

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)   # dropout

# 1. w1 = tf.Variable(tf.random_normal([784, 512], name = 'bias'))
# 2. w1 = tf.get_variable("w1", shape=[784, 512])
# 1과 2는 같은 뜻

# layer 1
w1 = tf.get_variable("w1", shape=[784, 512], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)


# layer 2
w2 = tf.get_variable("w2", shape=[512, 512], 
                     initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

# layer 3
w3 = tf.get_variable("w3", shape=[512, 512], 
                     initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

# layer 4
w4 = tf.get_variable("w4", shape=[512, 256], 
                     initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

# layer 5

w5 = tf.get_variable("w5", shape=[256, 10], 
                     initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis) , axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session() as sess:
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
print("ACC:", sess.run(accuracy, feed_dict = {x:x_test, y:y_test, keep_prob: 1}))