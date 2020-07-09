# 다중분류

from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = load_iris()

x_data = dataset.data
y_data = dataset.target

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name= 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_data = sess.run(tf.one_hot(y_data, depth = 3))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size =0.8,random_state= 10)

    
    for step in range(100):
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_train, y:y_train})

        if step % 10 ==0:
            print(step, cost_val)

    # 최적의 W와 b가 구해져 있다
        y_pred = sess.run(hypothesis, feed_dict={x:x_test})
        print(y_pred, sess.run(tf.argmax(y_pred, 1) + 1))
        
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_data, y: y_data}))

'''
 [1.2401874e-03 3.2569903e-01 6.7306083e-01]] 
 [2 3 1 2 1 2 2 2 1 3 3 3 2 1 1 3 2 1 1 1 3 3 2 1 2 1 2 3 2 3]
Accuracy :  0.9
'''