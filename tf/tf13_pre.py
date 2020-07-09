# preprocessing

from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)
    # 데이터를 넣어서 최솟값읋 찾는다. axis = 0 열에서 최솟값을 찾을꺼야!
    denominator = np.max(dataset, 0 ) - np.min(dataset, 0)

    return numerator / (denominator + 1e-7)
    # denominator에 + 1e-7 이유는 为了防止它成零

dataset = np.array(

    [

        [828.659973, 833.450012, 908100, 828.349976, 831.659973],

        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],

        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],

        [816, 820.958984, 1008100, 815.48999, 819.23999],

        [819.359985, 823, 1188100, 818.469971, 818.97998],

        [819, 823, 1198100, 816, 820.450012],

        [811.700012, 815.25, 1098100, 809.780029, 813.669983],

        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],

    ]

)

ds = min_max_scaler(dataset)
# print(ds)

x_d = dataset[:, 0:-1]     # (8, 4)
y_d = dataset[:, [-1]]     # (8, 1)
y_d1 = dataset[:, -1]      # (8, )

# x, y, w, b, hypothsis, cost, train(optimizer)

x_train, x_test, y_train, y_test = train_test_split(x_d, y_d, train_size = 0.8)

x = tf.placeholder(tf.float32, shape = [None, 4])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([4,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(h - y))

train = tf.train.GradientDescentOptimizer(learning_rate = 1e-20).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1001):
        loss_val, h_val, _ = sess.run([loss, h, train], feed_dict = {x: x_d, y: y_d})

        if i %10 == 1:
            print("횟수: ", i, "\n","loss: ", loss_val, "\n", "h_val: ", h_val)
    
    h_val = sess.run([h],feed_dict={x:x_test})

    # r2 = sess.run(r2_score(h_val,y_test))
    # print(r2)


