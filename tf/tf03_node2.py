import tensorflow as tf

# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2
node0 = tf.constant(2, tf.int32)
node1 = tf.constant(3, tf.int32)
node2 = tf.constant(4)
node3 = tf.constant(5)

# print(node1)
# print(node2)
# print(node3)

s = tf.Session()

# add = 덧셈
# add_n = list로 여러개를 받아서 계산 

node_add = tf.add_n([node1, node2, node3])

# subtract = 빼기
node_sub = tf.subtract(node2, node1)

# multiply = 곱셈
# matmul = 곱셈 두 가지. 행렬의 곱

node_mul = tf.multiply(node1, node2)

# divide = 나눗셈

node_dive = tf.divide(node2, node0)

print("3 + 4 + 5: ", s.run(node_add))
print("4 - 3: ", s.run(node_sub))
print("3 * 4: ", s.run(node_mul))
print("4 / 2: ", s.run(node_dive))

# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2