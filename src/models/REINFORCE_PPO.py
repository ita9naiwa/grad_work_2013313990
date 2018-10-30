import tensorflow as tf
import tflearn
import numpy as np

class model(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, network_widths=[300, 200, 30]):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.network_widths = network_widths

        self.states = tf.placeholder(tf.float32, [None, action_dim], 'state')
        self.actions = tf.placeholder(tf.int32, [None, ], 'action')
        self.advantages = tf.placeholder(tf.float32, [None, 1], 'advantage')

        self.out, self.params = self._create_network('new')
        self.old_out, old_params = self._create_network('old')

        self.update_old_op = [oldp.assign(p) for p, oldp in zip(self.params, old_params)]


        self.sess.run(tf.global_variables_initializer())


    def _create_network(self, name):
        #is it automatically flattend?
        #otherwise, manually flat inputs
        w_init = tflearn.initializations.xavier()
        b_init = tflearn.initializations.zeros()
        with tf.variable_scope(name):
            net = tflearn.fully_connected(self.states, self.network_widths[0], weights_init=w_init, bias_init=b_init)
            #net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            for network_width in self.network_widths[1:]:
                net = tflearn.fully_connected(net, network_width, weights_init=w_init)
                net = tflearn.layers.normalization.batch_normalization(net)
                net = tflearn.activations.relu(net)

            out = tflearn.fully_connected(net, self.action_dim)
            out = tflearn.activations.softmax(out)
        parameters = tf.trainable_variables(name)

        optimizer = (tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients, parameters)))

        return out, parameters
        # Supervised Learning

    def get_action_dist(self, state):
        return self.get_action([state])[0]

    def get_action(self, states):
        return self.sess.run(self.out,
                feed_dict={self.state_holder: states})

    def train(self, states, actions, values):
        [self.loss, self.optimizer]
        ret =  self.sess.run([self.indices, self.optimizer],
                feed_dict={
                    self.state_holder: states,
                    self.actions_holder: actions,
                    self.values_holder: values,
                    self.N: len(actions)})[0]
        #print(ret)
        return ret