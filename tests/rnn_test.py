import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell, LSTMCell
import numpy as np
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length



max_length = 100
frame_size = 64
num_hidden = 200

sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
output, state = tf.nn.dynamic_rnn(
    LSTMCell(num_hidden),
    sequence,
    dtype=tf.float32,
    sequence_length=length(sequence),
)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

ret = np.zeros(shape=(10, max_length, frame_size))
for i in range(10):
    mj = np.random.randint(max_length // 2, max_length)
    for j in range(mj):
        p = np.random.randint(0, frame_size)
        ret[i, j, p] = 1



a, b = sess.run([output, state], feed_dict={sequence: ret})
