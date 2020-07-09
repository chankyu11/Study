import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

# x, y, w, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy 준비

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w1 = tf.Variable(tf.random_normal([2, 100], name = 'weight1'))
b1 = tf.Variable(tf.random_normal([100], name = 'bias1'))
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
#  = model.add(Dense(3, input_shape = 2))

w2 = tf.Variable(tf.random_normal([100, 50]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([50], name = 'bias2'))
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
#  = model.add(Dense(3, input_shape = 1))


w3 = tf.Variable(tf.random_normal([50, 1]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([1], name = 'bias3'))
h = tf.sigmoid(tf.matmul(layer2, w3) + b3)

cost = -tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.009).minimize(cost)

predicted = tf.cast(h > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype= tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})
        print(i, "cost: ", cost_val)

    h, c, a = sess.run([h, predicted, accuracy], feed_dict = {x:x_data, y:y_data})    
    print("\n Hypothesis: ", h, "\n Correct (y): ", c,
          "\n Accuracy : ", a)
