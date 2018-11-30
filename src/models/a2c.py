import time
import functools
import tensorflow as tf

from baselines import logger
from pointer_network import pointer_networks
from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses

class actor(object):
    def __init__(self,
            state_size, input_size, embedding_size,
            hidden_size, max_job_length,
            nonlinear_transform=False,
            glimpse=False):

        super(actor, self).__init__()

        self.max_job_length = max_job_length
        self.state_size = state_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nonlinear_transform = nonlinear_transform
        self.glimpse = glimpse

        self.pointer_net = pointer_networks(
            self.state_size, self.input_size, self.embedding_size, self.hidden_size,
            self.max_job_length, self.nonlinear_transform, self.glimpse)
        



class Model(object):

    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn()