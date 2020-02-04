import tensorflow as tf
import algos.gail.core as core
import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import os
import os.path as osp
import pickle
from gym.spaces import Box, Discrete


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ReplayBuffer:
    def __init__(self, size, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.state = []
        self.action = []
        self.v = np.zeros((self.size, 1), np.float32)
        self.reward = np.zeros((self.size, 1), np.float32)
        self.adv = np.zeros((self.size, 1), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, v):
        if self.ptr < self.size:
            self.state.append(s)
            self.action.append(a)
            self.v[self.ptr] = v
            self.ptr += 1

    def finish_path(self, last_v):
        path_slice = slice(self.path_start, self.ptr)
        cur_obss = self.state[path_slice]
        cur_as = self.action[path_slice]
        reward = -np.log(np.clip(1-discriminator([np.stack(cur_obss, axis=0), np.stack(cur_as, axis=0)]), 1e-10, 1))

        v = np.stack(self.v[path_slice], axis=0)
        v_ = np.append(self.v[self.path_start + 1:self.ptr], last_v)
        adv = reward + self.gamma * v_[:, np.newaxis] - v
        adv = discount_cumsum(adv, self.gamma * self.lam)

        reward[-1] = reward[-1] + self.gamma * last_v
        self.reward[path_slice] = discount_cumsum(reward, self.gamma)

        self.adv[path_slice] = adv
        self.path_start = self.ptr

    def get(self):
        self.adv = (self.adv - np.mean(self.adv))/np.std(self.adv)
        return np.stack(self.state, axis=0), np.stack(self.action, axis=0), self.v, self.reward, self.adv


class ReplayBufferReward:
    def __init__(self, size, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.state = []
        self.action = []
        self.v = np.zeros((self.size, 1), np.float32)
        self.d = np.zeros((self.size, 1), np.int)
        self.reward = np.zeros((self.size, 1), np.float32)
        self.adv = np.zeros((self.size, 1), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, v, d):
        if self.ptr < self.size:
            self.state.append(s)
            self.action.append(a)
            self.v[self.ptr] = v
            self.d[self.ptr] = d
            self.ptr += 1

    def get(self):
        state = np.stack(self.state, axis=0)
        action = np.stack(self.action, axis=0)
        reward = -np.log(np.clip(1-discriminator([state, action]), 1e-10, 1))

        running_return = 0
        previous_value = 0
        running_adv = 0
        for i in reversed(range(len(reward))):
            running_return = reward[i] + self.gamma*self.d[i]*running_return
            self.reward[i] = running_return

            running_delta = reward[i] + self.gamma*self.d[i]*previous_value - self.v[i]
            previous_value = self.v[i]

            running_adv = running_delta + self.gamma*self.lam*self.d[i]*running_adv
            self.adv[i] = running_adv

        self.adv = (self.adv - np.mean(self.adv))/np.std(self.adv)
        return np.stack(self.state, axis=0), np.stack(self.action, axis=0), self.v, self.reward, self.adv

    def getSA(self):
        return np.stack(self.state, axis=0), np.stack(self.action, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=50)
    parser.add_argument('--d_update', type=int, default=1)
    parser.add_argument('--lr_a', default=4e-4)
    parser.add_argument('--lr_c', default=1e-3)
    parser.add_argument('--steps', default=1000)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--env', type=str, default="HalfCheetah-v2")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    file_path = osp.join(osp.abspath(osp.dirname(__file__)), "expert_data")
    file = osp.join(file_path, args.env + ".pkl")
    with open(file, "rb") as f:
        data = pickle.load(f)

    # expert_observations = np.genfromtxt('trajectory/observations.csv')
    # expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    expert_observations = data["observations"]
    expert_actions = data["actions"]
    expert_actions = np.squeeze(expert_actions)

    env = gym.make(args.env)
    tf.random.set_seed(1)
    np.random.seed(1)
    env.seed(1)
    action_space = env.action_space
    if isinstance(action_space, Discrete):
        ppo = core.PPO(action_space.n, 0.2, lr_a=args.lr_a, lr_c=args.lr_c)
    else:
        ppo = core.PPO(action_space.shape[0], 0.2, False, action_space.high[0], lr_a=args.lr_a, lr_c=args.lr_c)
    discriminator = core.Discriminator()
    opti_d = tf.keras.optimizers.Adam(0.0002, 0.5)

    summary_path = osp.join(osp.abspath(osp.dirname(__file__)), args.log)
    summary = tf.summary.create_file_writer(os.path.join(summary_path, "gail_" + args.env))
    replay = ReplayBufferReward(args.steps)

    s = env.reset()
    ppo.actor(s[np.newaxis, :])
    ppo.old_actor(s[np.newaxis, :])

    for iter in range(args.iteration):

        replay.reset()
        rewards = []
        obs = env.reset()
        rew = 0
        for step in range(args.steps):

            a = ppo.actor.select_action(obs[np.newaxis, :])
            if isinstance(action_space, Discrete):
                obs_, r, done, _ = env.step(a.numpy()[0])
            else:
                obs_, r, done, _ = env.step(a.numpy())
            rew += r
            v_pred = ppo.get_v(obs[np.newaxis, :])
            # print("step:", step, "a:", a, "r:", r, "rew:", rew, "done:", done, "obs:", obs[0:4])
            # replay.add(obs, a, v_pred)
            #
            # if done or step == args.steps-1:
            #     if done:
            #         replay.finish_path(np.array([0], np.float32))
            #     else:
            #         last_v = ppo.get_v(obs_[np.newaxis, :])
            #         replay.finish_path(last_v)
            #
            #     rewards.append(rew)
            #     rew = 0
            #     obs = env.reset()
            if done:
                replay.add(obs, a, v_pred, np.array([0]))
                rewards.append(rew)
                rew = 0
                obs = env.reset()
            else:
                replay.add(obs, a, v_pred, np.array([1]))
                obs = obs_

        with summary.as_default():
            tf.summary.scalar("reward", np.mean(rewards), iter)
        print("iter:", iter, "rewards:", np.mean(rewards))

        agent_s, agent_a = replay.getSA()
        agent = [agent_s, agent_a]
        for i in range(args.d_update):
            sample_indices = np.random.choice(range(np.shape(expert_observations)[0]), size=args.steps)
            expert_s = expert_observations[sample_indices]
            expert_a = expert_actions[sample_indices]
            # expert_a = expert_a[:, np.newaxis]
            expert = [expert_s, expert_a]

            with tf.GradientTape() as tape:
                expert_d = discriminator(expert)
                agent_d = discriminator(agent)

                loss_expert = tf.reduce_mean(tf.math.log(expert_d))
                loss_agent = tf.reduce_mean(tf.math.log(1-agent_d))
                loss = loss_agent + loss_expert
                loss = -loss

            grad = tape.gradient(loss, discriminator.trainable_variables)
            opti_d.apply_gradients(zip(grad, discriminator.trainable_variables))

        with summary.as_default():
            tf.summary.scalar("loss_d", loss, iter)

        agent_s, agent_a, agent_v, agent_r, agent_adv = replay.get()
        agent = [agent_s, agent_a]
        ppo.update_a()
        for i in range(args.a_update):
            aloss, logpi, old_logpi, mu, sigma, old_mu, old_sigma = ppo.train_a(agent_s, agent_a, agent_adv)

        for i in range(args.c_update):
            vloss, v = ppo.train_v(agent_s, agent_r)

        with summary.as_default():
            tf.summary.scalar("loss_a", aloss, iter)
            tf.summary.scalar("loss_v", vloss, iter)

            tf.summary.histogram("logpi", logpi, iter)
            tf.summary.histogram("logpi_old", old_logpi, iter)
            tf.summary.histogram("v", v, iter)
            tf.summary.histogram("vs", agent_r, iter)

            tf.summary.histogram("a", agent_a, iter)
            tf.summary.scalar("kl", tf.reduce_mean(old_logpi-logpi), iter)











