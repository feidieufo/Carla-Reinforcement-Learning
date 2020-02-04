import tensorflow as tf
from tensorflow.python.keras import layers
from algos.dqn.core import ICRA_Dqn
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

from env.carla_environment import CarlaEnvironmentWrapper as CarlaEnv

debug_mode = True

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
        state = {"img": [], "speed": [], "direction": []}
        action = []
        state_ = {"img": [], "speed": [], "direction": []}
        reward = []
        done = []
        for i in ids:
            (s, a, s_, r, d) = self.storage[i]
            state["img"].append(s["img"])
            state["speed"].append(s["speed"])
            state["direction"].append(s["direction"])

            action.append(a)

            state_["img"].append(s_["img"])
            state_["speed"].append(s_["speed"])
            state_["direction"].append(s_["direction"])

            reward.append(r)
            done.append(d)
        state["img"] = np.array(state["img"]).astype(np.float32)/255.0
        state["speed"] = np.expand_dims(np.array(state["speed"]), axis=1)
        state["direction"] = np.array(state["direction"])
        action = np.array(action)
        state_["img"] = np.array(state_["img"]).astype(np.float32)/255.0
        state_["speed"] = np.expand_dims(np.array(state_["speed"]), axis=1)
        state_["direction"] = np.array(state_["direction"])
        reward = np.array(reward)
        done = np.array(done)

        return state, action, state_, reward, done


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    step_left = decay_period + warmup_steps - step
    ep = step_left/decay_period*(1-epsilon)
    ep = np.clip(ep, 0, 1-epsilon)

    return epsilon + ep


class Dqn:
    def __init__(self, env_name, port=2000, gpu=0, batch_size=100, train_step=25000, evaluation_step=3000,
                 max_ep_len=6000, epsilon_train=0.1, epsilon_eval=0.01, replay_size=100000,
                 epsilon_decay_period=25000, warmup_steps=2000, iteration=200, gamma=0.99, q_lr=0.0001,
                 target_update_period=800, update_period=4, logger_kwargs=dict()):

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.env = CarlaEnv(early_termination_enabled=True, run_offscreen=False, port=port, gpu=gpu)

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

        self.build_model()
        self.cur_train_step = 0
        self.cur_tensorboard = 0

        if debug_mode:
            self.summary = tf.summary.create_file_writer(os.path.join(self.logger.output_dir, "logs"))

        self.build_model()
        self.savepath = os.path.join(self.logger.output_dir, "saver")
        checkpoint = tf.train.Checkpoint(model=self.model, target_model=self.model_target)
        self.manager = tf.train.CheckpointManager(checkpoint, directory=self.savepath, max_to_keep=20, checkpoint_name="model.ckpt")
        self.opti_q = tf.keras.optimizers.Adam(q_lr)


    def build_model(self):
        self.model = ICRA_Dqn(self.env.action_space.n)
        self.model_target = ICRA_Dqn(self.env.action_space.n)

    def choose_action(self, s, eval_mode=False):
        epsilon = self.epsilon_eval if eval_mode \
            else linearly_decaying_epsilon(self.epsilon_decay_period, self.cur_train_step, self.warmup_steps, self.epsilon_train)

        img = s["img"].astype(np.float32)/255.0
        speed = np.array([s["speed"]])
        if random.random() <= 1-epsilon:
            q = self.model([img[np.newaxis, :], speed[np.newaxis, :]])
            direction = s["direction"]
            q = q[direction]
            a = np.argmax(q, axis=1)[0]
        else:
            a = self.env.action_space.sample()

        return a

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

                s = obs
                a = self.choose_action(s, eval_mode)

                obs_, r, done, _ = self.env.step(a)

                step += 1
                step_episode += 1
                reward += r
                reward_episode += r

                if not eval_mode:
                    self.cur_train_step += 1
                    self.replay_buffer.add(obs, a, obs_, r, done)

                    if self.cur_train_step > self.warmup_steps:
                        if self.cur_train_step % self.update_period == 0:
                            (s, a, s_, r, d) = self.replay_buffer.sample(self.batch_size)

                            with tf.GradientTape() as tape:
                                img_ = s_["img"]                    # [None, shape]
                                speed_ = s_["speed"]                # [None, 1]
                                q_list = self.model_target([img_, speed_])
                                q_ = tf.stack(q_list[0:4], axis=1)       # [None, 4, 9]
                                direction_ = s_["direction"]
                                direction_ = tf.stack([tf.range(self.batch_size), direction_], axis=1)     # [None, 2]
                                q_ = tf.gather_nd(q_, direction_)                                      # [None, 9]

                                q_ = tf.reduce_max(q_, axis=1)
                                q_target = r + (1-d)*self.gamma * q_

                                img = s["img"]
                                speed = s["speed"]
                                qlist = self.model([img, speed])
                                q = tf.stack(qlist[0:4], axis=1)
                                direction = s["direction"]
                                direction = tf.stack([tf.range(self.batch_size), direction], axis=1)
                                q = tf.gather_nd(q, direction)                                            # [None, 9]

                                a_hot = tf.one_hot(a, depth=self.env.action_space.n)
                                q_pre = tf.reduce_sum(q * a_hot, axis=1)

                                def huber_loss(x, delta=1.0):
                                    # https://en.wikipedia.org/wiki/Huber_loss
                                    return tf.where(
                                        tf.abs(x) < delta,
                                        tf.square(x) * 0.5,
                                        delta * (tf.abs(x) - 0.5 * delta)
                                    )

                                # q_loss = tf.reduce_mean(huber_loss(q_pre - q_target))
                                q_loss = tf.reduce_mean(tf.square(q_pre - q_target))

                                # v_pred = q_list[4]
                                # v_loss = tf.reduce_mean(tf.square(v_pred - speed))
                                #
                                # loss = q_loss + v_loss

                            var = self.model.trainable_variables
                            q_gradient = tape.gradient(q_loss, self.model.trainable_variables)
                            # gra_var = [(gra, var) for gra, var in zip(q_gradient, self.model.trainable_variables) if gra!=None]
                            self.opti_q.apply_gradients(zip(q_gradient, self.model.trainable_variables))
                            # self.opti_q.apply_gradients(gra_var)

                        if self.cur_train_step % self.target_update_period == 0:
                            self.model_target.set_weights(self.model.get_weights())

                if step_episode >= self.max_ep_len:
                    break
                obs = obs_

            episode += 1
            if debug_mode and not eval_mode and self.cur_train_step > self.warmup_steps:
                self.cur_tensorboard += 1
                with self.summary.as_default():
                    tensorboard_step = self.cur_tensorboard
                    tf.summary.histogram("q_", q_, tensorboard_step)
                    tf.summary.histogram("a", a, tensorboard_step)
                    tf.summary.scalar("loss_q", q_loss, tensorboard_step)
                    # tf.summary.scalar("loss_v", v_loss, tensorboard_step)
                    # tf.summary.scalar("loss", loss, tensorboard_step)

                    # for var in self.model.trainable_variables:
                    #     tf.summary.histogram(var.name, var, tensorboard_step)
                    #
                    # for var in self.model_target.trainable_variables:
                    #     tf.summary.histogram(var.name, var, tensorboard_step)

            if not eval_mode:
                self.manager.save()



            # info = psutil.virtual_memory()

            # sys.stdout.write("steps: {}".format(step) + " episode_length: {}".format(step_episode) +
            #                  " return: {}".format(reward_episode) +
            #                  "  memory used : {}".format(psutil.Process(os.getpid()).memory_info().rss) +
            #                  " total memory: {}\r".format(info.total))
            #
            # sys.stdout.flush()

            print("ep:", episode, "step:", step_episode, "r:", reward_episode)
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
            # tf.logging.info("reward: %.2f, episode: %.2f", reward/episode, episode)

            reward, episode = self.run_one_phrase(self.evaluation_step, True)
            # print("reward:", reward / episode, "episode:", episode)
            # tf.logging.info("reward_test: %.2f, episode_test: %.2f", reward/episode, episode)

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
    parser.add_argument('--exp_name', type=str, default='dqn_benchmark_cil_onetwo')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn = Dqn(args.env, args.port, args.gpu, logger_kwargs=logger_kwargs)
    dqn.train_test()


