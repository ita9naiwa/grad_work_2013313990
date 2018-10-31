import tensorflow as tf
import tflearn
import numpy as np

class model(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate,
            network_widths=[300, 200, 30], update_step=30, epsilon=0.2):

        if sess is not None:
            self.sess = sess
        else:
            sess = tf.Session()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.network_widths = network_widths
        self.update_step = update_step
        self.epsilon = 0.2

        self.states = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.actions = tf.placeholder(tf.int32, [None, ], 'action')
        self.advantages = tf.placeholder(tf.float32, [None, ], 'advantage')

        self.out, self.params = self._create_network('new')
        self.old_out, old_params = self._create_network('old')

        self.update_old_op = [oldp.assign(p) for p, oldp in zip(self.params, old_params)]

        a_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)

        pi = tf.gather_nd(self.out, a_indices)
        oldpi = tf.gather_nd(self.old_out, a_indices)

        ratio = pi / oldpi

        surr = ratio * self.advantages
        self.loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.advantages))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
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
                #net = tflearn.layers.normalization.batch_normalization(net)
                net = tflearn.activations.relu(net)

            out = tflearn.fully_connected(net, self.action_dim)
            out = tflearn.activations.softmax(out)
        parameters = tf.trainable_variables(name)
        return out, parameters
        # Supervised Learning
    def get_one_act_prob(self, state):
        return self.get_action_dist(state)

    def get_action_dist(self, state):
        return self.get_action([state])[0]

    def get_action(self, states):
        return self.sess.run(self.out,
                feed_dict={self.states: states})

    def train(self, states, actions, values):
        s, a, adv = states, actions, values
        self.sess.run(self.update_old_op)
        losses = []
        for _ in range(self.update_step):
            loss = self.sess.run([self.loss, self.train_op],
                feed_dict={self.states: s, self.actions: a, self.advantages: adv})[0]
            losses.append(loss)
        print("temporal loss: %0.2f" % np.mean(losses))
        return losses
