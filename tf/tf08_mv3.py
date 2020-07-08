import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

ds = np.loadtxt('D:/STUDY/data/CSV/data-01-test-score.csv', delimiter = ',', dtype = np.float32)

x_data = ds[:,0:-1]
y_data = ds[:,[-1]]

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])
 
W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1], name = 'bias'))

hypothsis = tf.matmul(x,W) + b
# (5,3) * (3,1) = shape(5,1)

cost = tf.reduce_mean(tf.square(hypothsis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothsis, train],
                                feed_dict = {x: x_data, y: y_data})
    if step % 10 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)
