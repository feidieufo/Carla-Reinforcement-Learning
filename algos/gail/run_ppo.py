import tensorflow as tf
try:
    import algos.gail.core as core
except Exception:
    import core
import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import os
import os.path as osp
from gym.spaces import Box, Discrete
from utils.logx import EpochLogger


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

    def add(self, s, a, v, r):
        if self.ptr < self.size:
            self.state.append(s)
            self.action.append(a)
            self.v[self.ptr] = v
            self.reward[self.ptr] = r
            self.ptr += 1

    def finish_path(self, last_v):
        path_slice = slice(self.path_start, self.ptr)
        reward = self.reward[path_slice]

        v = np.stack(self.v[path_slice], axis=0)
        v_ = np.append(self.v[self.path_start+1:self.ptr], last_v)
        adv = reward + self.gamma*v_[:, np.newaxis] - v
        adv = discount_cumsum(adv, self.gamma * self.lam)

        reward[-1] = reward[-1] + self.gamma * last_v
        reward = discount_cumsum(reward, self.gamma)
        self.reward[path_slice] = reward

        self.adv[path_slice] = adv
        self.path_start = self.ptr

    def get(self):
        self.adv = (self.adv - np.mean(self.adv))/np.std(self.adv)
        return np.stack(self.state, axis=0), np.stack(self.action, axis=0), self.v, self.reward, self.adv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1000)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=50)
    parser.add_argument('--lr_a', default=3e-4)
    parser.add_argument('--lr_c', default=3e-4)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', type=int, default=1000)
    # parser.add_argument('--env', type=str, default="CartPole-v0")
    parser.add_argument('--v_gae_clip', default=False)
    parser.add_argument('--env', type=str, default="HalfCheetah-v2")
    parser.add_argument('--exp_name', type=str, default="orthogonal")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from utils.run_utils import setup_logger_kwargs
    file_name = "ppo_" + args.env + "_" + args.exp_name
    logger_kwargs = setup_logger_kwargs(file_name, args.seed)
    logger = EpochLogger(**logger_kwargs)

    env = gym.make(args.env)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    action_space = env.action_space
    if isinstance(action_space, Discrete):
        ppo = core.PPO(action_space.n, 0.2, lr_a=args.lr_a, lr_c=args.lr_c)
    else:
        ppo = core.PPO(action_space.shape[0], 0.2, False, action_space.high[0], lr_a=args.lr_a, lr_c=args.lr_c)

    # summary_file = osp.join(osp.abspath(osp.dirname(__file__)), args.log)
    # file_name = "ppo_" + args.env + "_" + args.exp_name + "_" + str(args.seed)
    summary = tf.summary.create_file_writer(os.path.join(logger.output_dir, args.log))
    saver_file = osp.join(logger.output_dir, "saver")
    # saver_file = osp.join(saver_file, file_name)
    checkpoint = tf.train.Checkpoint(actor=ppo.actor, old_actor=ppo.old_actor, critic=ppo.critic)

    replay = ReplayBuffer(args.steps)

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
            replay.add(obs, a, v_pred, r)

            # print("step:", step, "a:", a, "r:", r, "rew:", rew, "done:", done, "obs:", obs[0:4])

            obs = obs_
            if done or step == args.steps-1:
                if done:
                    replay.finish_path(np.array([0], np.float32))
 
                else:
                    last_v = ppo.get_v(obs_[np.newaxis, :])
                    replay.finish_path(last_v)

                logger.store(rewards=rew)
                rewards.append(rew)
                rew = 0
                obs = env.reset()

        with summary.as_default():
            tf.summary.scalar("reward", np.mean(rewards), iter)
        # print("iter:", iter, "rewards:", np.mean(rewards))
        print("iter:", iter+1)
        logger.log_tabular("rewards", with_min_and_max=True)
        logger.dump_tabular()

        agent_s, agent_a, agent_v, agent_r, agent_adv = replay.get()

        ppo.update_a()
        lr = (1-iter/args.iteration)*args.lr_a
        for i in range(args.a_update):
            aloss, logpi, old_logpi, mu, sigma, old_mu, old_sigma = ppo.train_a(agent_s, agent_a, agent_adv, lr)
            # aloss, logpi, old_logpi = ppo.train_a(agent_s, agent_a, agent_adv)

        if args.v_gae_clip:
            oldvs = ppo.critic(agent_s)
            for i in range(args.c_update):
                vloss, v = ppo.train_v_gae_loss(agent_s, oldvs, agent_adv, lr)
        else:
            for i in range(args.c_update):
                vloss, v = ppo.train_v(agent_s, agent_r, lr)

        checkpoint.save(osp.join(saver_file, "model.ckpt"))

        with summary.as_default():
            tf.summary.scalar("loss_a", aloss, iter)
            tf.summary.scalar("loss_v", vloss, iter)
            tf.summary.histogram("v", v, iter)
            tf.summary.histogram("vs", agent_r, iter)
            tf.summary.histogram("logpi", logpi, iter)
            tf.summary.histogram("logpi_old", old_logpi, iter)

            tf.summary.histogram("mu", mu, iter)
            tf.summary.histogram("mu_old", old_mu, iter)
            tf.summary.histogram("sigma", sigma, iter)
            tf.summary.histogram("sigma_old", old_sigma, iter)

            tf.summary.histogram("agent_a", agent_a, iter)

            tf.summary.scalar("kl", tf.reduce_mean(old_logpi-logpi), iter)













