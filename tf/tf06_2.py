# tf06_1.py를 복사
# lr을 수정해서 연습
# 0,01 -> 0.1 / 0.001/ 1
# epoch가 2000번을 적게 만들어라

import tensorflow as tf

tf.set_random_seed(777)

# x_train = [1,2,3]
# y_train = [1,2,3]

x_train = tf.compat.v1.placeholder(tf.float32, shape = [None])
y_train = tf.compat.v1.placeholder(tf.float32, shape = [None])
# compat.v1.placeholder = compat 안에 v1안에 placeholder이 존재

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1], name = 'bias'))

sess = tf.Session()

sess.run(tf.global_variables_initializer())
# 초기화 변수는 초기화 후 진행해야함

print(sess.run(W))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 1e-10).minimize(cost)

with tf.Session() as sess:
# with tf.compat.v1.Session as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.compat.v1.global_variables_initializer())
    # 변수 선언이라 할 수 있음.
    # 한번 돌아가고 초기화하고 돌아가고 초기화하고

    for step in range(1981):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],
                                    feed_dict = {x_train: [1,2,3], y_train:[3,5,7]})

        if step % 20 ==0:
            print(step, cost_val, W_val, b_val)
    # predict해보기
    print(sess.run(hypothesis, feed_dict = {x_train:[4]}))
    
'''
lr = 1

1900 nan [nan] [nan]
1920 nan [nan] [nan]
1940 nan [nan] [nan]
1960 nan [nan] [nan]
1980 nan [nan] [nan]

lr = 1e-10
1900 9.57655 [0.80269563] [0.45847914]
1920 9.57655 [0.80269563] [0.45847914]
1940 9.57655 [0.80269563] [0.45847914]
1960 9.57655 [0.80269563] [0.45847914]
1980 9.57655 [0.80269563] [0.45847914]
[3.6692617]

'''
