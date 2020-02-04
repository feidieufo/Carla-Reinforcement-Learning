import numpy as np
import tensorflow as tf
import gym
import time
from algos.td3 import core
from algos.td3.core import get_vars
from utils.logx import EpochLogger

from env.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
import os

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

        batch = {"obs1": np.array(state).astype(np.float32)/255.0,
                "obs2": np.array(state_).astype(np.float32)/255.0,
                "acts": np.array(action),
                "rews": np.array(reward),
                "done": np.array(done)}
        return batch

class Td3:
    def __init__(self, env_name, port=2000, gpu=0, train_step=25000, evaluation_step=3000, max_ep_len=6000, alpha=0.35,
                 epsilon_train=0.1, polyak=0.995, start_steps=1000, batch_size=100, replay_size=50000,
                 iteration=200, gamma=0.99, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
                 pi_lr=1e-4, q_lr=1e-4, policy_delay=2,
                target_update_period=800, update_period=4, logger_kwargs=dict()):

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.iteration = iteration
        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.env = CarlaEnv(early_termination_enabled=True, run_offscreen=False, port=port, gpu=gpu, discrete_control=False)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.alpha = alpha
        self.start_steps = start_steps
        self.cur_train_step = 0
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.act_limit = self.env.action_space.high[0]
        self.act_noise = act_noise
        self.policy_delay = policy_delay

        if debug_mode:
            self.summary = tf.summary.FileWriter(os.path.join(self.logger.output_dir, "logs"))

        # self.obs_dim = (30, 30, 3)
        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
            core.placeholders(self.obs_dim, self.act_dim, self.obs_dim, None, None)

        self.actor_critic = core.ActorCritic(self.act_dim)
        self.target_actor_critic = core.ActorCritic(self.act_dim)
        self.q1, self.q2, self.pi = self.actor_critic([self.x_ph, self.a_ph])
        self.q1_pi, _, _ = self.actor_critic([self.x_ph, self.pi])

        tar_q1, tar_q2, _ = self.target_actor_critic([self.x_ph, self.a_ph])
        _, _, pi_targ = self.target_actor_critic([self.x2_ph, self.a_ph])
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -self.act_limit, self.act_limit)
        q1_targ, q2_targ, _ = self.target_actor_critic([self.x2_ph, a2])

        # Main outputs from computation graph
        # with tf.variable_scope('main'):
        #     # self.pi, self.q1, self.q2, q1_pi = core.cnn_actor_critic(self.x_ph, self.a_ph)
        #     self.pi, self.q1, = core.cnn_actor_critic(self.x_ph, self.a_ph)
        #     self.q2, q1_pi = self.pi, self.q1

        # # Target policy network
        # with tf.variable_scope('target'):
        #     pi_targ, _, _, _  = core.cnn_actor_critic(self.x2_ph, self.a_ph)
        #
        #
        # # Target Q networks
        # with tf.variable_scope('target', reuse=True):
        #     # Target policy smoothing, by adding clipped noise to target actions
        #     epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        #     epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        #     a2 = pi_targ + epsilon
        #     a2 = tf.clip_by_value(a2, -self.act_limit, self.act_limit)
        #
        #     # Target Q-values, using action from target policy
        #     _, q1_targ, q2_targ, _ = core.cnn_actor_critic(self.x2_ph, a2)
        # q1_targ, q2_targ = q1_pi, q1_pi
        # q3, q4 = q1_pi, q1_pi

        self.replay_buffer = ReplayBuffer(replay_size)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*min_q_targ)

        # TD3 losses
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        q1_loss = tf.reduce_mean((self.q1-backup)**2)
        q2_loss = tf.reduce_mean((self.q2-backup)**2)
        self.q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        # self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi') + get_vars('main/cnn'))
        # self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=get_vars('main/q') + get_vars('main/cnn'))

        self.train_pi_op = pi_optimizer.minimize(self.pi_loss)
        self.train_q_op = q_optimizer.minimize(self.q_loss)

        # var = [v.name for v in tf.global_variables()]
        # v = tf.trainable_variables()
        var = [v.name for v in tf.trainable_variables()]
        print(var)

        if debug_mode:
            tf.summary.histogram("main/q1", self.q1)
            tf.summary.histogram("main/q2", self.q2)
            tf.summary.histogram("main/q1_pi", self.q1_pi)
            tf.summary.histogram("target/tar_q1", tar_q1)
            tf.summary.histogram("target/tar_q2", tar_q2)
            tf.summary.histogram("target/q1_tar", q1_targ)
            tf.summary.histogram("target/q2_tar", q2_targ)

            tf.summary.histogram("a/pi", self.pi)
            tf.summary.histogram("a/pi_tar", pi_targ)
            tf.summary.histogram("a/a2", a2)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            tf.summary.scalar("loss_q1", q1_loss)
            tf.summary.scalar("loss_q2", q2_loss)
            tf.summary.scalar("loss_q", self.q_loss)
            tf.summary.scalar("loss_pi", self.pi_loss)

            self.merge = tf.summary.merge_all()

        # Polyak averaging for target variables
        # self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
        #                           for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(self.actor_critic.trainable_variables,
                                                            self.target_actor_critic.trainable_variables)])

        # Initializing targets to match main variables
        # target_init = tf.group([tf.assign(v_targ, v_main)
        #                           for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(self.actor_critic.trainable_variables,
                                                            self.target_actor_critic.trainable_variables)])

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session("", config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)
        self.saver = tf.train.Checkpoint(model=self.actor_critic)
        self.savepath = os.path.join(self.logger.output_dir, "saver")
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.manager = tf.train.CheckpointManager(self.saver, self.savepath, max_to_keep=10)

    def get_action(self, o, noise_scale, eval_mode = False):
        o = np.array(o).astype(np.float32)/255.0
        a = self.sess.run(self.pi, feed_dict={self.x_ph: np.expand_dims(o, axis=0)})[0]
        # print("----ori:" + str(a))
        if not eval_mode:
            a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

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
                if self.cur_train_step > self.start_steps or eval_mode:
                    a = self.get_action(s, self.act_noise, eval_mode)
                else:
                    a = self.env.action_space.sample()

                # print(a)
                obs_, r, done, _ = self.env.step(a)

                step += 1
                step_episode += 1
                reward += r
                reward_episode += r

                if not eval_mode:
                    self.cur_train_step += 1
                    self.replay_buffer.add(obs, a, obs_, r, done)

                if step_episode >= self.max_ep_len:
                    break
                obs = obs_

            episode += 1
            if self.cur_train_step > self.start_steps and not eval_mode:
                for j in range(step_episode):
                    batch = self.replay_buffer.sample(self.batch_size)
                    feed_dict = {self.x_ph: batch['obs1'],
                                 self.x2_ph: batch['obs2'],
                                 self.a_ph: batch['acts'],
                                 self.r_ph: batch['rews'],
                                 self.d_ph: batch['done']
                                }
                    q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
                    outs = self.sess.run(q_step_ops, feed_dict)
                    self.logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                    if debug_mode and (self.cur_train_step-(step_episode-j)):
                            merge = self.sess.run(self.merge, feed_dict=feed_dict)
                            self.summary.add_summary(merge, (self.cur_train_step-(step_episode-j)))

                    if j % self.policy_delay == 0:
                        # Delayed policy update
                        outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
                        self.logger.store(LossPi=outs[0])

            if episode % 10 == 0:
                # self.manager.save()
                # self.saver.save(os.path.join(self.savepath, "model.ckpt"), self.sess)
                self.actor_critic.save_weights(os.path.join(self.savepath, "model_" + str(episode)))

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
            self.logger.log_tabular('Q1Vals', with_min_and_max=True)
            self.logger.log_tabular('Q2Vals', with_min_and_max=True)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossQ', average_only=True)
            self.logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='carla')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3_carla')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    td3 = Td3(args.env, port=args.port, gpu=args.gpu, logger_kwargs=logger_kwargs)
    td3.train_test()
