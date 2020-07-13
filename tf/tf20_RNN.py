import tensorflow as tf
import numpy as np

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h','i','h','e','l','l','o']]).T
# print(_data.shape)
# print(_data)
# print(type(_data))


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
# print("_data: \n", _data)
'''
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''
"======================================================="
x_data = _data[:6, ]
# print("x_data: \n", x_data)
'''
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]]
'''
"======================================================="
y_data = _data[1:, ]
# print("y_data\n", y_data)
'''
 [[0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''
"======================================================="

y_data = np.argmax(y_data, axis=1)
# print(y_data)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6,)

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

# print(X)
# print(Y)

# 2. 모델구성

# cell = tf.nn.rnn_cell.BasicLSTMCell(ouput)
# model.add(LSTM(output, input_shape = (6,5)))

cell = tf.keras.layers.LSTMCell(output)
h, _states = tf.nn.dynamic_rnn(cell, X , dtype = tf.float32)

# print("h: ", h)
# print("states:", _states)
'''
h:  Tensor("rnn/transpose_1:0", shape=(?, 6, 100), dtype=float32)
states: LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 100) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 100) dtype=float32>)
'''
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = h, targets = Y, weights = weights)

cost = tf.reduce_mean(sequence_loss)

# train = tf.train.AdadeltaOptimizer(learning_rate = 0.1).minimize(loss)
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate = 0.1).minimize(cost)

predict = tf.argmax(h, axis = 2)

# 3-2 훈련

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(401):
        loss = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
        result = sess.run(predict, feed_dict = {X:x_data})
        print(i, "loss: ", loss, "predict: ", result, "true_y: ", y_data)
        
        # result_str = [idx2char[c] for c in np.squeeze(result)]
        # print("\nPrediction str: ", ''.join(result_str))
        
        result_str = [idx2char[c] for c in np.squeeze(result)]  # np.sqeeze() : 찾아보기
        print("\nPrediction str : ", ''.join(result_str))       # ''.join() : 붙인다, 찾아보기


