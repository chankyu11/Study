# 이진분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np


x1, y1 = load_breast_cancer(return_X_y= True)
print('x',x1.shape, 'y',y1.shape)

y1 = y1.reshape(569, 1)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothsis = tf.sigmoid(tf.matmul(x,w) + b)

cost = tf.reduce_mean(y * tf.log(hypothsis) + (1 - y) * tf.log(1 - hypothsis))
train = tf.train.GradientDescentOptimizer(learning_rate= 1e-10).minimize(cost)


predicted = tf.cast(hypothsis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype= tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x1, y:y1})
        print(i, "cost: ", cost_val)

    h, c, a = sess.run([hypothsis, predicted, accuracy], feed_dict = {x:x1, y:y1})    
    print("\n Hypothesis: ", h, "\n Correct (y): ", c,
          "\n Accuracy : ", a)


'''
 Accuracy :  0.6274165
'''