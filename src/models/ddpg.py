import tensorflow as tf
import tflearn
import numpy as np

class DDPG(object):
    def __init__(self, sess, action_dim, input_size, actor_learning_rate=0.0001, critic_learning_rate=0.001, tau=0.001):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.discount = 0.9
        self.target_critic_update = None
        self.target_actor_update = None
        self.input_size = input_size
        self.action_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau

        self.actor = self.create_actor_network("actor")
        self.target_actor = self.create_actor_network("target_actor")

        self.critic = self.create_critic_network("critic")
        self.target_critic = self.create_critic_network("target_critic")

        # calc critic loss
        self.y_i = tflearn.input_data(shape=(None, 1))
        self.critic_loss = tf.reduce_mean(tf.square(tf.subtract(self.critic['out'], self.y_i)))
        self.optimizer_critic_loss = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.critic_loss)

        # gradient w.r.t action of critic
        self.action_grads = tf.gradients(self.critic['out'], self.critic['actions'])
        self.action_grads_holder = tflearn.input_data(shape=(None, self.action_dim))

        # gradient w.r.t ?
        self.actor_grads = tf.gradients(self.actor['out'], self.actor['params'], -self.action_grads_holder)
        self.optimizer_actor_loss = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(
            zip(self.actor_grads, self.actor['params']))

        #target update
        param = self.critic['params']
        target_param = self.target_critic['params']
        self.target_critic_update = [target_param[i].assign(
                tf.multiply(param[i], self.tau) + tf.multiply(target_param[i], 1. - self.tau))
                for i in range(len(param))]

        param = self.actor['params']
        target_param = self.target_actor['params']
        self.target_actor_update = [target_param[i].assign(
                tf.multiply(param[i], self.tau) + tf.multiply(target_param[i], 1. - self.tau))
                for i in range(len(param))]


    def get_q(self, critic, states, actions):
        return self.sess.run(critic['out'],
            feed_dict = {
                critic['states']: states,
                critic['actions']: actions})

    def get_action_dist(self, state):
        return self.get_action(self.actor, [state])[0]

    def get_action(self, actor, states):
        dist_given_state = self.sess.run(
            actor['out'],
            feed_dict={
                actor['states']: states})
        return dist_given_state

    def train(self, samples):
        pred = self.critic_step(samples)
        self.actor_step(samples)
        self.update_target_params()
        return pred

    def actor_step(self, samples):
        s_i, a_i, r_i, t_i, s_j = samples
        actions = self.get_action(self.actor, s_i)
        action_grads = self.sess.run(
            self.action_grads,
            feed_dict={
                self.critic['states']: s_i,
                self.critic['actions']: actions
            })[0]

        self.sess.run(self.optimizer_actor_loss,
            feed_dict={
                self.actor['states']: s_i,
                self.action_grads_holder: action_grads
            })
        return action_grads

    def critic_step(self, samples):
        batch_size = len(samples[0])
        s_i, a_i, r_i, t_i, s_j = samples
        a_j_pred = self.get_action(self.target_actor, s_j)
        target_q_j = self.get_q(self.target_critic, s_j, a_j_pred)
        y_i = np.zeros(shape=(batch_size, 1), dtype=np.float32)
        for k in range(batch_size):
            if t_i[k] is True:
                y_i[k, 0] = r_i
            else:
                y_i[k, 0] = r_i[k] + self.discount * target_q_j[k,0]

        #print(a_i)
        pred_q, _ = self.sess.run(
            [self.critic['out'], self.optimizer_critic_loss],
            feed_dict={
                self.y_i: y_i,
                self.critic['states']: s_i,
                self.critic['actions']: a_i
            }
        )
        #print(np.array(critic_loss[0]))
        return pred_q

    def update_target_params(self):
        self.sess.run(self.target_critic_update)
        self.sess.run(self.target_actor_update)

    def create_critic_network(self, namescope):

        w_init = tflearn.initializations.truncated_normal(stddev=0.01)
        with tf.variable_scope(namescope):
            states = inputs = tflearn.input_data(shape=(None, self.input_size), dtype=tf.float32, name="states")
            actions = tflearn.input_data(shape=(None, self.action_dim), dtype=tf.float32, name="actions")
            net_l = tflearn.fully_connected(inputs, 300, weights_init=w_init)
            #net_l = tflearn.layers.normalization.batch_normalization(net_l)
            net_l = tflearn.activations.relu(net_l)
            net_a = tflearn.fully_connected(actions, 300, weights_init=w_init)
            net_a = tflearn.activations.relu(net_a)

            net = tflearn.layers.merge_ops.merge([net_l, net_a], mode='concat')
            net = tflearn.fully_connected(net, 300, weights_init=w_init)
            net = tflearn.activations.relu(net)
            o = tflearn.fully_connected(net, 1)

        parameters = tf.trainable_variables(namescope)
        print("len of critic net parameters: %d" % len(parameters))
        for p in parameters:
            print(p)
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
            net = tflearn.fully_connected(inputs, 300, weights_init=w_init)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, 300, weights_init=w_init)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            out = tflearn.fully_connected(net, self.action_dim, weights_init=w_init)
            out = tflearn.activations.tanh(out)

        parameters = tf.trainable_variables(namescope)
        print("len of actor net parameters: %d" % len(parameters))
        for p in parameters:
            print(p)

        return {
            "states": states,
            "out": out,
            "params": parameters,
        }


def main(args):
    #3actor_network(object):
    #ef __init__(self, sess, action_dim, input_width, input_height, learning_rate, tau):

    with tf.Session() as sess:
        DDPG(sess, 1, input_widths)

if __name__ == "__main__":
    main()