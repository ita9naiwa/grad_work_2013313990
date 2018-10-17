import tensorflow as tf
import tflearn
import numpy as np


def nd_flatten(ndarr, d=1):
    shape = ndarr.shape
    newshape = tuple(list(shape[:d]) + [np.prod(shape[d:])])
    newarray = np.reshape(ndarr, newshape)
    return newarray


class actor_network(object):
    def __init__(self, sess, action_dim, input_width, input_height, learning_rate, tau):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.input_width, self.input_height = input_width, input_height
        self.input_size = self.input_width * self.input_height
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.network_widths = [22]
        self.learning_rate
        (self.out, self.state_holder, self.actions_holder, self.values_holder,
        self.loss, self.optimizer) = self._create_network()
        self.sess.run(tf.global_variables_initializer())


    def _create_network(self):
        #is it automatically flattend?
        #otherwise, manually flat inputs
        w_init = tflearn.initializations.normal(stddev=0.01)
        actions = tflearn.input_data(shape=(None,), dtype=tf.int32)
        values = tflearn.input_data(shape=(None,), dtype=tf.float32)
        states = inputs = tflearn.input_data(shape=(None, self.input_size), dtype=tf.float32)
        net = tflearn.fully_connected(inputs, self.network_widths[0], weights_init=w_init)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        out = tflearn.fully_connected(net, self.action_dim, activation='softmax', weights_init=w_init)
        N = tf.shape(inputs)[0]

        parameters = tf.trainable_variables()



        # Policy Gradient
        o = tf.gather_nd(out, tf.stack([tf.range(N), actions], axis=-1))
        loss = tf.reduce_mean(tf.tensordot(tf.log(o), values, 1))
        gradidents = tf.gradients(loss, parameters)
        optimizer = (tf.train.RMSPropOptimizer(self.learning_rate).
                     apply_gradients(
                     zip(gradidents, parameters)))
        return out, states, actions, values, loss, optimizer
        # Supervised Learning

    def p_given_state(self, observations):
        shape = observations.shape
        if len(shape) == 2:
            new_ret = np.reshape(observations, (1, np.prod(shape, dtype=int)))
        elif len(shape) == 3:
            new_ret = np.reshape(observations, (shape[0], np.prod(shape[:-2], dtype=int)))

        return self.sess.run(self.out, {self.state_holder: new_ret})

    def choose_action(self, observations):
        action_dist = self.p_given_state(observations)
        return np.argmax(action_dist)

    def train(self, states, actions, values):
        return self.sess.run([self.out, self.optimizer],
            feed_dict={
                self.state_holder: states,
                self.actions_holder: actions,
                self.values_holder: values})[0]
