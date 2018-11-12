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
        self.states = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.actions = tf.placeholder(tf.int32, [None, ], 'action')
        self.advantages = tf.placeholder(tf.float32, [None, ], 'advantage')

        self.out, self.optimizer = self._create_network()
        self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        #is it automatically flattend?
        #otherwise, manually flat inputs
        w_init = tflearn.initializations.xavier()
        b_init = tflearn.initializations.zeros()
        states = self.states
        net = tflearn.fully_connected(states, self.network_widths[0], weights_init=w_init, bias_init=b_init)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        for network_width in self.network_widths[1:]:
            net = tflearn.fully_connected(net, network_width, weights_init=w_init)
            #net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, self.action_dim)
        #if use_softmax = False:
        out = tflearn.activations.softmax(out)

        # Policy Gradient

        a_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
        self.p_s_a = p_s_a = tf.gather_nd(out, a_indices)
        self.loss = loss = -tf.reduce_mean(p_s_a * self.advantages)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        #gradients = [-g for g in gradients]
        optimizer = (tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients, parameters)))



        return out, optimizer
        # Supervised Learning

    def get_action_dist(self, state):
        return self.get_action([state])[0]

    def get_action(self, states):
        return self.sess.run(self.out,
                feed_dict={self.states: states})

    def train(self, states, actions, values):
        ret = self.sess.run([self.loss, self.optimizer],
                feed_dict={
                    self.states: states,
                    self.actions: actions,
                    self.advantages: values})[0]
        #print(ret)
        return ret