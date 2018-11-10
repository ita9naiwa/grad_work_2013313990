import tensorflow as tf
import tflearn
import numpy as np
from tensorflow.nn.rnn_cell import GRUCell, LSTMCell

class model(object):
    def __init__(
            self, sess, machine_shape, seq_length, seq_size, action_dim, learning_rate,
            network_widths=[300, 200, 30]):
            #use_softmax=False):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()

        self.machine_shape = machine_shape
        self.seq_length = seq_length
        self.seq_size = seq_size

        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.network_widths = network_widths
        self.learning_rate
        (self.out, self.state_holder, self.actions_holder, self.values_holder,
        self.loss, self.optimizer) = self._create_network()
        self.sess.run(tf.global_variables_initializer())


    def _create_network(self):
        #is it automatically flattend?
        #otherwise, manually flat inputs
        w_init = tflearn.initializations.normal(mean=0.0, stddev=0.01)
        b_init = tflearn.initializations.zeros()
        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length

        self.input_machine = tf.placeholder(tf.float32, [None, self.machine_shape])
        self.input_seq = tf.placeholder(tf.float32, [None, self.seq_length, self.seq_size])
        machine_net = tflearn.fully_connected(self.input_machine, 20, weights_init=w_init)

        seq_net, _ = tf.nn.dynamic_rnn(
            LSTMCell(20), self.input_seq,
            dtype=tf.float32)#, sequence_length=length(self.input_seq))
        seq_net = tf.contrib.layers.flatten(seq_net)
        net = tflearn.layers.merge_ops.merge([machine_net, seq_net], mode='concat')
        net = tflearn.fully_connected(net, 20, weights_init=w_init, bias_init=b_init)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        for network_width in self.network_widths[1:]:
            net = tflearn.fully_connected(net, network_width, weights_init=w_init)
            #net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, self.action_dim)
        #if use_softmax = False:
        out = tflearn.activations.softmax(out)
        parameters = tf.trainable_variables()

        # Policy Gradient
        actions = tflearn.input_data(shape=(None,), dtype=tf.int32, name='actions')
        values = tflearn.input_data(shape=(None,), dtype=tf.float32, name='values')
        self.N = N = tflearn.input_data(shape=(), dtype=tf.int32)
        self.indices = indices = tf.stack([tf.range(N, dtype=tf.int32), actions], axis=1)
        self.p_s_a = p_s_a = tf.gather_nd(out, indices)
        self.loss = loss = -tf.reduce_mean(p_s_a * values)
        gradients = tf.gradients(loss, parameters)
        #gradients = [-g for g in gradients]
        optimizer = (tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients, parameters)))

        parameters = tf.trainable_variables()

        return out, None, actions, values, loss, optimizer
        # Supervised Learning


    def get_action(self, machine_repr, seq):
        return self.sess.run(self.out,
                feed_dict={
                    self.input_machine: [machine_repr],
                    self.input_seq: [seq],
                })[0]

    def train(self, states, actions, values):
        ret = self.sess.run([self.loss, self.optimizer],
                feed_dict={
                    self.input_machine: states[0],
                    self.input_seq: states[1],
                    self.actions_holder: actions,
                    self.values_holder: values,
                    self.N: len(actions)})[0]
        #print(ret)
        return ret