import tensorflow as tf
import numpy as np

x_data =[[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,6,7]]
y_data =[[0,1,1],
         [0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name= 'bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        _, cost_val = sess.run([optimizer, cost],
                                feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
        a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
        print(a, sess.run(tf.arg_max(a, 1)))
        
        b = sess.run(hypothesis, feed_dict={x: [[11, 33, 4, 13]]})
        print(b, sess.run(tf.arg_max(b, 1)))

        c = sess.run(hypothesis, feed_dict={x: [[1, 1, 0, 1]]})
        print(c, sess.run(tf.arg_max(c, 1)))

        all = sess.run(hypothesis, feed_dict = {x:[[1,11,7,9], [11, 33, 4, 13],[1, 1, 0, 1]]})
        print(all, sess.run(tf.arg_max(all, 1)))