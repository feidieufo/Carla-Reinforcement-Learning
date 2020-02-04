import tensorflow as tf
try:
    import algos.gail.core_cnn as core
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

from env.carla_environment import CarlaEnvironmentWrapper as CarlaEnv
import os
from user_config import DEFAULT_DATA_DIR
from utils.logx import EpochLogger

debug_mode = True


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
        self.state = {"img": [], "speed": [], "direction": []}
        self.action = []
        self.v = np.zeros((self.size, 1), np.float32)
        self.reward = np.zeros((self.size, 1), np.float32)
        self.adv = np.zeros((self.size, 1), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, v, r):
        if self.ptr < self.size:
            self.state["img"].append(s["img"])
            self.state["speed"].append(s["speed"])
            self.state["direction"].append(s["direction"])
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

        self.state["img"] = np.array(self.state["img"]).astype(np.float32)/255.0
        self.state["speed"] = np.expand_dims(np.array(self.state["speed"]), axis=1)
        self.state["direction"] = np.array(self.state["direction"], dtype=np.int32)

        return self.state, np.stack(self.action, axis=0), self.v, self.reward, self.adv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=50)
    parser.add_argument('--lr_a', default=4e-4)
    parser.add_argument('--lr_c', default=1e-3)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=300)
    parser.add_argument('--port', default=2000)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--exp_name', default="ppo_carla")
    parser.add_argument('--seed', default=0)
    parser.add_argument('--batch', default=50)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    env = CarlaEnv(early_termination_enabled=True, run_offscreen=False, port=args.port, gpu=args.gpu, discrete_control=False)
    ppo = core.PPO(3, 0.2, lr_a=args.lr_a, lr_c=args.lr_c)

    if debug_mode:
        summary = tf.summary.create_file_writer(os.path.join(logger.output_dir, "logs"))

    savepath = osp.join(logger.output_dir, "saver")
    checkpoint = tf.train.Checkpoint(model=ppo)
    manger = tf.train.CheckpointManager(checkpoint, directory=savepath, max_to_keep=20, checkpoint_name="model.ckpt")
    replay = ReplayBuffer(args.steps)

    s = env.reset()
    img = s["img"].astype(np.float32) / 255.0
    speed = np.array([s["speed"]])
    c = s["direction"]
    ppo.actor([img[np.newaxis, :], speed[np.newaxis, :]])
    ppo.old_actor([img[np.newaxis, :], speed[np.newaxis, :]])

    train_a_loss = tf.keras.metrics.Mean(name='train_d_loss')
    train_c_loss = tf.keras.metrics.Mean(name='train_g_loss')

    for iter in range(args.iteration):

        replay.reset()
        rewards = []
        obs = env.reset()
        rew = 0
        for step in range(args.steps):
            img = obs["img"].astype(np.float32) / 255.0
            speed = np.array([obs["speed"]])
            c = obs["direction"]

            a = ppo.actor.select_action([img[np.newaxis, :], speed[np.newaxis, :]], c)

            obs_, r, done, _ = env.step(a)
            rew += r
            v_pred = ppo.getV({"img": img[np.newaxis, :], "speed": speed[np.newaxis, :], "direction": c})
            replay.add(obs, a, v_pred, r)

            obs = obs_
            if done or step == args.steps-1:
                if done:
                    replay.finish_path(np.array([0], np.float32))
                else:
                    img = obs_["img"].astype(np.float32) / 255.0
                    speed = np.array([obs_["speed"]])
                    c = obs_["direction"]
                    last_v = ppo.getV({"img": img[np.newaxis, :], "speed": speed[np.newaxis, :], "direction": c})
                    replay.finish_path(last_v)

                rewards.append(rew)
                logger.store(reward=rew)
                rew = 0
                obs = env.reset()

        if debug_mode:
            with summary.as_default():
                tf.summary.scalar("reward", np.mean(rewards), iter)
        # print("iter:", iter, "rewards:", np.mean(rewards))
        logger.log_tabular("reward", with_min_and_max=True)
        logger.dump_tabular()

        agent_s, agent_a, agent_v, agent_r, agent_adv = replay.get()
        data = tf.data.Dataset.from_tensor_slices((agent_s, agent_a, agent_r, agent_adv))
        data = data.shuffle(100).batch(args.batch)

        ppo.update_a()
        train_a_loss.reset_states()
        train_c_loss.reset_states()

        for i in range(args.a_update):
            for (s, a, r, adv) in data:
                aloss, logpi, old_logpi, mu, sigma, old_mu, old_sigma = ppo.train_a(s, a, adv)
                train_a_loss(aloss)
            # aloss, logpi, old_logpi = ppo.train_a(agent_s, agent_a, agent_adv)
        print("aloss", train_a_loss.result())

        for i in range(args.c_update):
            for (s, a, r, adv) in data:
                vloss, v = ppo.train_v(s, r)
                train_c_loss(vloss)
        print("closs", train_c_loss.result())

        manger.save()

        if debug_mode:
            with summary.as_default():
                tf.summary.scalar("loss_a", train_a_loss.result(), iter)
                tf.summary.scalar("loss_v", train_c_loss.result(), iter)
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













