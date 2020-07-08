from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np

## 데이터 불러오기

x1, y1 = load_diabetes(return_X_y= True)
print('x',x1.shape, 'y',y1.shape)   # (442, 10) # (442,)

y1 = y1.reshape(442,1)

x = tf.placeholder(dtype = tf.float32, shape=[None, 10])
y = tf.placeholder(dtype = tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name='weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name='bias', dtype = tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, w, b], feed_dict = {x: x1, y: y1})

        if step % 20 ==0:
            print(step, cost_val, W_val, b_val)