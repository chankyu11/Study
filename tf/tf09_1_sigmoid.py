import tensorflow as tf

tf.set_random_seed(777)

x_d = [[1, 2],
       [2, 3],
       [3, 1],
       [4, 3],
       [5, 3],
       [6, 2]]

y_d = [[0],
       [0],
       [0],
       [1],
       [1],
       [1]]        

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])
 
W = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1], name = 'bias'))

hypothsis = tf.sigmoid(tf.matmul(x,W) + b)
# (5,3) * (3,1) = shape(5,1)

cost = tf.reduce_mean(y * tf.log(hypothsis) + (1 - y) * tf.log(1 - hypothsis))
# sigmoid cost 모르겠다면 그냥 암기

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-10)

train = optimizer.minimize(cost)

predicted = tf.cast(hypothsis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype= tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(3001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_d, y:y_d})
        print(i, "cost: ", cost_val)

    h, c, a = sess.run([hypothsis, predicted, accuracy], feed_dict = {x:x_d, y:y_d})
    print("\n Hypothesis: ", h, "\n Correct (y): ", c,
          "\n Accuracy : ", a)