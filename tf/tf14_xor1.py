import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

# x, y, w, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy 준비

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.zeros([2, 1], name = 'weight'))
b = tf.Variable(tf.zeros([1], name = 'bias'))

h = tf.sigmoid(tf.matmul(x,w) + b)

cost = tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h))

train = tf.train.GradientDescentOptimizer(learning_rate = 1e-10).minimize(cost)

predicted = tf.cast(h > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype= tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})
        print(i, "cost: ", cost_val)

    h, c, a = sess.run([h, predicted, accuracy], feed_dict = {x:x_data, y:y_data})    
    print("\n Hypothesis: ", h, "\n Correct (y): ", c,
          "\n Accuracy : ", a)
