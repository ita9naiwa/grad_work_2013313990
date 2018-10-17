import tensorflow as tf
import tflearn
import numpy as np


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

        self.critic = self.create_critic_network("critic")
        self.target_critic = self.create_critic_network("target_critic")

        self.actor = self.create_actor_network("actor")
        self.target_actor = self.create_actor_network("target_actor")


        self.sess.run(tf.global_variables_initializer())

    def update_target_params(self):

        target_critic = [
            target_param.assign(param * tau + (1.0 - tau) * target_param)
            for param, target_param in
            zip(self.critic['params'], self.target_critic['params'])
        ]

        target_actor = [
            target_param.assign(param * tau + (1.0 - tau) * target_param)
            for param, target_param in
            zip(self.actor['params'], self.actor['params'])
        ]

    def create_critic_network(self, namescope):
        w_init = tflearn.initializations.normal(stddev=0.01)

        with tf.variable_scrpe(namescope):
            states = inputs = tflearn.input_data(shape=(None, self.input_size) dtype=tf.float32)
            actions = tflearn.input_data(shape=(None, self.action_dim), dtype=tf.float32)
            net_i = tflearn.fully_connected(inputs, 400, weights_init=w_init)
            net_l = tflearn.layers.normalization.batch_normalization(net_l, 300)
            #net_l = tflearn.activations.relu(net_l, 200)
            net_a = tflearn.fully_connected(inputs, 300, weights_init=w_init)
            #net_a = tflearn.activations.relu(net_a)
            net = tflearn.layers.merge_ops.merge([net_l, net_a], mode='concat', axis=1)
            net = tflearn.fully_connected(net, 300, weights_init=w_init)
            net = tflearn.activations.relu(net)
            o = tflearn.fully_connected(net, 1)

        parameters = tf.trainable_variables(namescope)
        return {
            "states": states,
            "actions": actions,
            "out": o,
            "params": parameters
        }

    def create_actor_network(self, namescope):
        #is it automatically flattend?
        #otherwise, manually flat inputs
        with tf.variable_scope(namescope):
            w_init = tflearn.initializations.normal(stddev=0.01)
            states = inputs = tflearn.input_data(shape=(None, self.input_size), dtype=tf.float32)
            net = tflearn.fully_connected(inputs, self.network_widths[0], weights_init=w_init)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            out = tflearn.fully_connected(net, self.action_dim, activation='softmax', weights_init=w_init)
            N = tf.shape(inputs)[0]

        parameters = tf.trainable_variables(namescope)
        """
        o = tf.gather_nd(out, tf.stack([tf.range(N), actions], axis=-1))
        loss = tf.reduce_mean(tf.tensordot(tf.log(o), values, 1))
        gradidents = tf.gradients(loss, parameters)
        optimizer = (tf.train.RMSPropOptimizer(self.learning_rate).
                     apply_gradients(
                     zip(gradidents, parameters)))
        """

        return {
            "states": states,
            "out": out,
            "params"; parameters
        }



        # Policy Gradient

        return out, states, actions, values, loss, optimizer
        # Supervised Learning
