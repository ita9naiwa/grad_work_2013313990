import tensorflow as tf
import tflearn
import numpy as np

class model(object):
    def __init__(
            self, sess, state_dim, action_dim, learning_rate,
            network_widths=[300, 200, 30]):
            #use_softmax=False):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.state_dim = state_dim
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
        w_init = tflearn.initializations.xavier()
        b_init = tflearn.initializations.zeros()
        states = inputs = tflearn.input_data(shape=(None, self.state_dim), dtype=tf.float32, name="states")
        net = tflearn.fully_connected(inputs, self.network_widths[0], weights_init=w_init, bias_init=b_init)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        for network_width in self.network_widths[1:]:
            net = tflearn.fully_connected(net, network_width, weights_init=w_init)
            net = tflearn.layers.normalization.batch_normalization(net)
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

        return out, states, actions, values, loss, optimizer
        # Supervised Learning

    def get_action_dist(self, state):
        return self.get_action([state])[0]

    def get_action(self, states):
        return self.sess.run(self.out,
                feed_dict={self.state_holder: states})

    def train(self, states, actions, values):
        [self.loss, self.optimizer]
        ret =  self.sess.run([self.loss, self.optimizer],
                feed_dict={
                    self.state_holder: states,
                    self.actions_holder: actions,
                    self.values_holder: values,
                    self.N: len(actions)})[0]
        #print(ret)
        return ret