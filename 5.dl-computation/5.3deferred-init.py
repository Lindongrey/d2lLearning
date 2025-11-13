import tensorflow as tf

net = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

# 数据还未通过，权重没有初始化
print([net.layers[i].get_weights() for i in range(len(net.layers))])
X = tf.random.uniform((2, 20))
net(X)
# 在数据通过后，权重就初始化完成
print([w.shape for w in net.get_weights()])
