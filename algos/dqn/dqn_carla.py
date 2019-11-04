import tensorflow as tf
from tensorflow.python.keras import layers
from algos.dqn.core import nature_dqn
import numpy as np
import random
import copy
from utils.logx import EpochLogger
import os
import sys
import matplotlib.pyplot as plt
import cv2
import gym
import psutil
from tensorflow.python.keras.backend import set_session

from env.carla_environment_wrapper_docker import CarlaEnvironmentWrapper as CarlaEnv

debug_mode = False

class ReplayBuffer:
    def __init__(self, size):
        self.storage = []
        self.maxsize = size
        self.next_idx = 0
        self.size = 0

    def add(self, s, a, s_, r, done):
        data = (s, a, s_, r, done)
        if self.size < self.maxsize:
            self.storage.append(data)

        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.maxsize
        self.size = min(self.size+1, self.maxsize)

    def sample(self, batch_size=32):
        ids = np.random.randint(0, self.size, size=batch_size)
        # storage = np.array(self.storage)
        state = []
        action = []
        state_ = []
        reward = []
        done = []
        for i in ids:
            (s, a, s_, r, d) = self.storage[i]
            state.append(np.array(s, copy=False))
            action.append(a)
            state_.append(np.array(s_, copy=False))
            reward.append(r)
            done.append(d)

        return np.array(state).astype(np.float32)/255.0, np.array(action), np.array(state_).astype(np.float32)/255.0, np.array(reward), np.array(done)


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    step_left = decay_period + warmup_steps - step
    ep = step_left/decay_period*(1-epsilon)
    ep = np.clip(ep, 0, 1-epsilon)

    return epsilon + ep


class Dqn:
    def __init__(self, env_name, port=2000, gpu=0, batch_size=32, train_step=25000, evaluation_step=3000,
                 max_ep_len=6000, epsilon_train=0.1, epsilon_eval=0.01, replay_size=100000,
                 epsilon_decay_period=25000, warmup_steps=2000, iteration=200, gamma=0.99,
                 target_update_period=800, update_period=4, logger_kwargs=dict()):

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.env = CarlaEnv(early_termination_enabled=True, run_offscreen=True, port=port, gpu=gpu)

        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.max_ep_len = max_ep_len
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.epsilon_decay_period = epsilon_decay_period
        self.warmup_steps = warmup_steps
        self.iteration = iteration
        self.replay_buffer = ReplayBuffer(replay_size)
        self.gamma = gamma
        self.target_update_period = target_update_period
        self.update_period = update_period

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session("", config=config)
        set_session(self.sess)

        self.build_model()
        self.cur_train_step = 0

        self.observation_shape = (84, 84)
        self.state_shape = (1,) + self.observation_shape + (4,)
        self.s = np.zeros(self.state_shape)
        self.last_s = np.zeros(self.state_shape)

        if debug_mode:
            self.summary = tf.summary.FileWriter(os.path.join(self.logger.output_dir, "logs"))



        self.loss = tf.placeholder(tf.float32, shape=[])
        self.q = tf.placeholder(tf.float32, shape=[None, self.env.action_space.n])
        self.q_target = tf.placeholder(tf.float32, shape=[None, self.env.action_space.n])
        self.target_q = tf.placeholder(tf.float32, shape=[None, self.env.action_space.n])
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("q", self.q)
        # tf.summary.histogram("q_target", self.q_target)
        # tf.summary.histogram("target_q", self.target_q)
        self.merge = tf.summary.merge_all()

    def build_model(self):
        self.model, self.model_target = nature_dqn(self.env.action_space.n, (80, 80, 6))
        self.model_target.set_weights(self.model.get_weights())

    def choose_action(self, s, eval_mode=False):
        epsilon = self.epsilon_eval if eval_mode \
            else linearly_decaying_epsilon(self.epsilon_decay_period, self.cur_train_step, self.warmup_steps, self.epsilon_train)

        if random.random() <= 1-epsilon:
            q = self.model.predict(s[np.newaxis, :])
            a = np.argmax(q, axis=1)[0]
            # print()
        else:
            a = self.env.action_space.sample()

        return a

    def record_obs(self, observation):
        self.last_s = copy.copy(self.s)
        self.s = np.roll(self.s, -1, axis=-1)
        self.s[0, ..., -1] = np.squeeze(observation)

    def store(self, s, a, s_, r, done):
        pass

    def run_one_phrase(self, min_step, eval_mode=False):
        step = 0
        episode = 0
        reward = 0.

        while step < min_step:
            done = False
            obs = self.env.reset()

            step_episode = 0
            reward_episode = 0
            while not done:

                s = np.array(obs)
                a = self.choose_action(s, eval_mode)

                obs_, r, done, _ = self.env.step(a)

                step += 1
                step_episode += 1
                reward += r
                reward_episode += r
                
                if not eval_mode:
                    self.cur_train_step += 1
                    self.replay_buffer.add(obs, a, obs_, r, done)

                    if self.cur_train_step > 2000:
                        if self.cur_train_step % self.update_period == 0:
                            # data = self.replay_buffer.sample()
                            (s, a, s_, r, d) = self.replay_buffer.sample(self.batch_size)
                            q_ = np.max(self.model_target.predict(s_), axis=1)
                            q_target = r + (1-d)*self.gamma * q_
                            q = self.model.predict(s)
                            q_recoder = np.copy(q)

                            batch_index = np.arange(self.batch_size)
                            q[batch_index, a] = q_target
                            result = self.model.train_on_batch(np.array(s), q)
                            # print("result:", result)

                            # if self.cur_train_step%1== 0:
                            #     merge = self.sess.run(self.merge, feed_dict={self.loss: result[0], self.q: q_recoder})
                            #     self.summary.add_summary(merge, (self.cur_train_step-20000)/self.update_period/1)

                        if self.cur_train_step % self.target_update_period == 0:
                            self.model_target.set_weights(self.model.get_weights())

                if step_episode >= self.max_ep_len:
                    break
                obs = obs_

            episode += 1

            savepath = os.path.join(self.logger.output_dir, "saver")
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            self.model.save(os.path.join(savepath, "model" + str(episode%5) + ".h5"))
            # info = psutil.virtual_memory()

            # sys.stdout.write("steps: {}".format(step) + " episode_length: {}".format(step_episode) +
            #                  " return: {}".format(reward_episode) +
            #                  "  memory used : {}".format(psutil.Process(os.getpid()).memory_info().rss) +
            #                  " total memory: {}\r".format(info.total))
            #
            # sys.stdout.flush()

            print("ep:", episode, "step:", step, "r:", reward)
            if not eval_mode:
                self.logger.store(step=step_episode, reward=reward_episode)
            else:
                self.logger.store(step_test=step_episode, reward_test=reward_episode)

        return reward, episode

    def train_test(self):
        for i in range(self.iteration):
            print("iter:", i+1)
            self.logger.store(iter=i+1)
            reward, episode = self.run_one_phrase(self.train_step)
            # print("reward:", reward/episode, "episode:", episode)
            tf.logging.info("reward: %.2f, episode: %.2f", reward/episode, episode)

            reward, episode = self.run_one_phrase(self.evaluation_step, True)
            # print("reward:", reward / episode, "episode:", episode)
            tf.logging.info("reward_test: %.2f, episode_test: %.2f", reward/episode, episode)

            self.logger.log_tabular("reward", with_min_and_max=True)
            self.logger.log_tabular("step", with_min_and_max=True)
            self.logger.log_tabular("reward_test", with_min_and_max=True)
            self.logger.log_tabular("step_test", with_min_and_max=True)
            self.logger.dump_tabular()





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Carla')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--exp_name', type=str, default='dqn_carla_random_pos_replaybuffer10e5')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn = Dqn(args.env, args.port, args.gpu, logger_kwargs=logger_kwargs)
    dqn.train_test()


