## p.118 
## placeholder

import tensorflow as tf

# tf.placeholder(dtype, shape = ??, name = ??)
 
x = tf.placeholder("float")
y = 2 * x
data = tf.random_uniform([4,5], 10)
with tf.Session() as sess:
    x_data = sess.run(data)
    print(sess.run(y, feed_dict = {x:x_data})) 

