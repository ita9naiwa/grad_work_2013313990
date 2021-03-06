import tensorflow as tf
import tflearn
import numpy as np

class DDPG(object):
    def __init__(
        self, sess, action_dim, state_dim,
        discount=0.9,
        actor_learning_rate=0.0001, critic_learning_rate=0.001,
        tau=0.001, use_softmax=False):
        if sess != None:
            self.sess = sess
        else:
            sess = tf.Session()
        self.discount = 0.9
        self.target_critic_update = None
        self.target_actor_update = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        self.use_softmax = use_softmax

        self.actor = self.create_actor_network("actor")
        self.target_actor = self.create_actor_network("target_actor")

        self.critic = self.create_critic_network("critic")
        self.target_critic = self.create_critic_network("target_critic")

        # calc critic lossD
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

        #target sup_pred
        self.pred_p = self.actor['out']
        self.input_actions = tflearn.input_data(shape=(None, self.action_dim))
        self.sup_obj = tf.reduce_mean(tflearn.objectives.categorical_crossentropy(self.pred_p, self.input_actions))
        self.optimizer_sup_actor_loss = tf.train.AdamOptimizer(0.001).minimize(self.sup_obj)
    def train_actor_sup(self, states, target_actions):
        actions = np.zeros(shape=(states.shape[0], self.action_dim))
        for i, t in enumerate(target_actions):
            actions[i, t] = 1.0

        return self.sess.run([self.sup_obj, self.optimizer_sup_actor_loss],
            feed_dict={
                self.actor['states']: states,
                self.input_actions: actions})[0]


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
                y_i[k, 0] = r_i[k] + self.discount * target_q_j[k, 0]

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

        w_init = tflearn.initializations.xavier()
        with tf.variable_scope(namescope):
            states = inputs = tflearn.input_data(shape=(None, self.state_dim), dtype=tf.float32, name="states")
            actions = tflearn.input_data(shape=(None, self.action_dim), dtype=tf.float32, name="actions")
            net_l = tflearn.fully_connected(inputs, 50, weights_init=w_init)
            #net_l = tflearn.layers.normalization.batch_normalization(net_l)
            net_l = tflearn.activations.relu(net_l)
            net_a = tflearn.fully_connected(actions, 50, weights_init=w_init)
            #net_a = tflearn.fully_connected(net_a, 50, weights_init=w_init)
            #net_a = tflearn.layers.normalization.batch_normalization(net_a)
            net_a = tflearn.activations.relu(net_a)

            net = tflearn.layers.merge_ops.merge([net_l, net_a], mode='concat')
            #net = tflearn.fully_connected(net, 150, weights_init=w_init)
            #net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, 50, weights_init=w_init)
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
        w_init = tflearn.initializations.xavier()
        with tf.variable_scope(namescope):

            states = inputs = tflearn.input_data(shape=(None, self.state_dim), dtype=tf.float32)
            net = tflearn.fully_connected(inputs, 20, weights_init=w_init)
            #net = tflearn.layers.normalization.batch_normalization(net)
            #net = tflearn.activations.relu(net)
            #net = tflearn.fully_connected(net, 40, weights_init=w_init)
            net = tflearn.activations.relu(net)
            out = tflearn.fully_connected(net, self.action_dim, weights_init=w_init)
            if self.use_softmax:
                out = tflearn.activations.softmax(out)
            else:
                out = tflearn.activations.tanh(out)

        parameters = tf.trainable_variables(namescope)

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