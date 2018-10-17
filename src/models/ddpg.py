import tensorflow as tf
import tflearn
import numpy as np


class actor_network(object):
    def __init__(self, sess, action_dim, input_width, input_height, learning_rate, tau):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.discount = 1.0
        self.target_critic_update = None
        self.target_actor_update = None

        self.input_width, self.input_height = input_width, input_height
        self.input_size = self.input_width * self.input_height
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.network_widths = [22]
        self.learning_rate

        self.actor = self.create_actor_network("actor")
        self.target_actor = self.create_actor_network("target_actor")


        self.critic = self.create_critic_network("critic")
        self.target_critic = self.create_critic_network("target_critic")

        # calc critic loss
        self.y_i = tf.input_data(shape=(None, 1))
        self.critic_loss = tflearn.mean_square(self.critic['out'], self.y_i)
        self.optimizer_critic_loss = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

        # gradient w.r.t action of critic
        self.action_grads = tf.gradients(self.critic['out'], self.critic['actions'])
        self.actor_grads = tf.gradients(self.actor['out'],  self['actor']['params'], -self.action_grads)
        self.actor_grads = [ tf.div(x, self.batch_size) for x in self.actor_grads]

        self.optimizer_actor_loss = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_grads, self.actor['params']))

    def generate_action(self, actor, states):
        dist_given_state = self.sess.run(
            actor['out'],
            feed_dict={
                actor['states']: state
            }
        )
        return np.argmax(dist_given_state, axis=1)

    def critic_step(self, samples):
        s_i, a_i, r_i, s_j = samples
        a_j_pred = self.generate_action(self.target_actor, s_j)

        q_pred = self.sess.run(
            self.critic_target['out'],
            feed_dict={
                self.critic_target['states']: s_j,
                self.critic_target['actions']: a_j_pred
            }
        )
        print("critic output shape: ", r_i)
        y_i = r_i + self.discount * q_pred

        critic_loss = self.sess.run(
            [self.critic_loss, self.optimizer_critic_loss],
            feed_dict={
                self.y_i: y_i,
                self.critic['states']: s_i,
                self.critic['actons']: a_i
            }
        )[0]
        return critic_loss



    def update_target_params(self):

        if self.target_critic_update is None:
            self.target_critic_update = [
                target_param.assign(param * tau + (1.0 - tau) * target_param)
                for param, target_param in zip(self.critic['params'], self.target_critic['params'])]

        if self.target_actor_update is None:
            self.target_actor_update = [
                target_param.assign(param * tau + (1.0 - tau) * target_param)
                for param, target_param in zip(self.actor['params'], self.actor['params'])]

        self.sess.run(self.target_critic_update + self.target_actor_update)


    def create_critic_network(self, namescope):

        w_init = tflearn.initializations.normal(stddev=0.01)

        with tf.variable_scrpe(namescope):
            states = inputs = tflearn.input_data(shape=(None, self.input_size), dtype=tf.float32)
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
            "params": parameters,
        }



        # Policy Gradient

        return out, states, actions, values, loss, optimizer
        # Supervised Learning
