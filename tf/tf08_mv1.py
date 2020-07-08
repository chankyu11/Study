import tensorflow as tf

tf.set_random_seed(777)

x1_d = [74., 93., 89., 96., 73.]
x2_d = [80., 88., 91., 98., 66.]
x3_d = [75., 93., 90., 100., 70.]

y_d = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1], name = 'weight1'))
w2 = tf.Variable(tf.random_normal([1], name = 'weight1'))
w3 = tf.Variable(tf.random_normal([1], name = 'weight1'))
b = tf.Variable(tf.random_normal([1], name = 'weight1'))

hypothsis = x1 * w1 + x2 * w2 + x3 + w3 + b

cost = tf.reduce_mean(tf.square(hypothsis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000045)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothsis, train],
                                feed_dict = {x1: x1_d, x2: x2_d,
                                            x3: x3_d, y: y_d})
    if step % 10 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)
