import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1], name = 'bias'))

sess = tf.Session()
# print(W)

W = tf.Variable([0.3], tf.float32)
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa:",aaa)
sess.close()
# 세션 닫는거 명시 필요, 데이터가 크다면 엉길수가 있기에 닫아야함.
# 이거 안하려면 with 사용

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print("bbb:",bbb)
sess.close()
# 위와 같음
# InteractiveSession 은 eval()을 사용.
# sess = tf.Session을 쓰면 sess.run()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess)
print("ccc:",ccc)
sess.close()