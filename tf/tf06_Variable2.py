# hypothesis를 구하시오
# H = Wx + b

import tensorflow as tf
tf.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1], tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("hypothesis1 : ", )
sess.close()

# InteractiveSession()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print("hypothesis2 : ", )
sess.close()

# Session(), eval()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess)
print("hypothesis3 : ", )
sess.close()