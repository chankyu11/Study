import tensorflow as tf
import numpy as np

ds = np.array([1,2,3,4,5,6,7,8,9,10])
size = 5
# print(ds.shape)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(ds, size)
# print(dataset)

x_d = dataset[:, :-1]
y_d = dataset[:, 4:]

# print(x_d.shape)    # 6, 4 
# print(y_d.shape)    # 6, 1


x_d = x_d.reshape(1,6,4)
y_d = y_d.reshape(1,6)
# print(x_d)
# print(y_d)
# print(x_d.shape)   
# print(y_d.shape)    

X = tf.compat.v1.placeholder(tf.float32, shape = (None, 6, 4))
Y = tf.compat.v1.placeholder(tf.int64, shape = (None, 6))

_lstm = tf.keras.layers.LSTMCell(11)

h, _states = tf.nn.dynamic_rnn(_lstm, X, dtype = tf.float32)

weights = tf.compat.v1.ones([1, 6])       
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = h, targets = Y, weights = weights)

cost = tf.compat.v1.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)


# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict = {X: x_d, Y: y_d})
        # result = sess.run(prediction, feed_dict = {X: x_d})
        # print(f'\nEpoch : {i}, loss : {loss}')
        print(i, "loss: ", loss)
   