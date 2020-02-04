import numpy as np
import tensorflow as tf
import gym
import time
from algos.td3 import core
from utils.logx import EpochLogger

from env.carla_environment import CarlaEnvironmentWrapper as CarlaEnv
import os
from user_config import DEFAULT_DATA_DIR

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

        state["img"] = np.array(state["img"]).astype(np.float32)/255.0      # [?, 88, 200, 3]
        state["speed"] = np.expand_dims(np.array(state["speed"]), axis=1)   # [?, 1]
        state["direction"] = np.array(state["direction"])                   # [?, ]
        state_["img"] = np.array(state_["img"]).astype(np.float32)/255.0
        state_["speed"] = np.expand_dims(np.array(state_["speed"]), axis=1)
        state_["direction"] = np.array(state_["direction"])

        batch = {"obs1": state,
                "obs2": state_,
                "acts": np.array(action),
                "rews": np.array(reward),
                "done": np.array(done)}
        return batch


class Td3:
    def __init__(self, env_name, port=2000, gpu=0, train_step=10000, evaluation_step=3000, max_ep_len=300,
                 polyak=0.995, start_steps=200, batch_size=100, replay_size=50000,
                 iteration=200, gamma=0.99, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
                 pi_lr=1e-4, q_lr=1e-3, policy_delay=2, logger_kwargs=dict()):

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.iteration = iteration
        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.env = CarlaEnv(early_termination_enabled=True, run_offscreen=True, port=port, gpu=gpu, discrete_control=False)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.start_steps = start_steps
        self.cur_train_step = 0
        self.cur_tensorboard_step = 0
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.act_limit = self.env.action_space.high[0]
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.polyak = polyak
        self.gamma = gamma
        self.opti_q = tf.keras.optimizers.Adam(q_lr)
        self.opti_pi = tf.keras.optimizers.Adam(pi_lr)

        if debug_mode:
            self.summary = tf.summary.create_file_writer(os.path.join(self.logger.output_dir, "logs"))

        self.actor_critic = core.ActorCritic(self.act_dim, self.act_limit)
        self.target_actor_critic = core.ActorCritic(self.act_dim, self.act_limit)
        self.replay_buffer = ReplayBuffer(replay_size)

        self.loadpath = os.path.join(DEFAULT_DATA_DIR, "saver_0.45_0.45_0.05_0.1_tfaug_shuffle_first")
        actor = core.ActorCnn()
        load_check = tf.train.Checkpoint(model=actor)
        load_check.restore(os.path.join(self.loadpath, "model.ckpt-200"))

        # with tf.GradientTape() as tape:
        #     x = tf.random.uniform(minval=0, maxval=1, shape=self.obs_dim)
        #     x = tf.expand_dims(x, axis=0)
        #     a = tf.random.uniform(minval=0, maxval=1, shape=[self.act_dim])
        #     a = tf.expand_dims(a, axis=0)
        #     self.actor_critic([x,a])
        #     self.actor_critic.choose_action(x)
        #     self.target_actor_critic([x,a])
        #     self.target_actor_critic.choose_action(x)
        with tf.GradientTape() as tape:
            img = tf.random.uniform(minval=0, maxval=1, shape=self.obs_dim)
            img = tf.expand_dims(img, axis=0)
            speed = tf.random.uniform(minval=0, maxval=1, shape=(1,))
            speed = tf.expand_dims(speed, axis=0)
            self.actor_critic.actor([img, speed])
            self.target_actor_critic.actor([img, speed])
            actor([img, speed])
        for old_var, var in zip(actor.variables, self.actor_critic.variables):
            var.assign(old_var)
        var = self.actor_critic.actor.trainable_variables
        old_var = actor.trainable_variables

        self.target_init(self.target_actor_critic, self.actor_critic)

        self.savepath = os.path.join(self.logger.output_dir, "saver")
        checkpoint = tf.train.Checkpoint(model=self.actor_critic, target_model=self.target_actor_critic)
        self.manager = tf.train.CheckpointManager(checkpoint, directory=self.savepath, max_to_keep=20, checkpoint_name="model.ckpt")

    def get_action(self, o, noise_scale, eval_mode=False):
        img = o["img"].astype(np.float32)/255.0
        speed = np.array([o["speed"]])
        direction = o["direction"]

        a_list, z = self.actor_critic.select_action([img[np.newaxis, :], speed[np.newaxis, :]])
        a = tf.squeeze(a_list[direction], axis=0)    # [act_dim]
        # print("----ori:" + str(a))
        if not eval_mode:
            a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def target_init(self, target_net, net):
        for target_params, params in zip(target_net.trainable_variables, net.trainable_variables):
            target_params.assign(params)

    def target_update(self, target_net, net):
        for target_params, params in zip(target_net.trainable_variables, net.trainable_variables):
            target_params.assign(self.polyak*target_params+(1-self.polyak)*params)


    def train_q(self, batch):
        with tf.GradientTape() as tape:
            img1 = batch['obs1']["img"]
            speed1 = batch['obs1']["speed"]
            direction1 = batch['obs1']["direction"]
            direction1 = tf.stack([tf.range(self.batch_size), direction1], axis=1)    # [None, 2]

            img2 = batch["obs2"]["img"]
            speed2 = batch["obs2"]["speed"]
            direction2 = batch["obs2"]["direction"]
            direction2 = tf.stack([tf.range(self.batch_size), direction2], axis=1)  # [None, 2]

            q1_list, q2_list = self.actor_critic([img1, speed1, batch["acts"]])
            q1_list = tf.stack(q1_list, axis=1)     # [None, 4, 1]
            q2_list = tf.stack(q2_list, axis=1)     # [None, 4, 1]
            q1 = tf.gather_nd(q1_list, direction1)  # [None, 1]
            q2 = tf.gather_nd(q2_list, direction1)

            pi_targ_list, z = self.target_actor_critic.select_action([img2, speed2])
            pi_targ_list = tf.stack(pi_targ_list[0:4], axis=1)     # [None, 4, 3]
            pi_targ = tf.gather_nd(pi_targ_list, direction2)  # [None, 3]

            epsilon = tf.random.normal(tf.shape(pi_targ), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -self.act_limit, self.act_limit)

            q1_targ_list, q2_targ_list = self.target_actor_critic([img2, speed2, a2])
            q1_targ_list = tf.stack(q1_targ_list, axis=1)     # [None, 4, 1]
            q2_targ_list = tf.stack(q2_targ_list, axis=1)     # [None, 4, 1]
            q1_targ = tf.gather_nd(q1_targ_list, direction2)  # [None, 1]
            q2_targ = tf.gather_nd(q2_targ_list, direction2)
            min_q_targ = tf.minimum(q1_targ, q2_targ)
            backup = batch['rews'] + self.gamma*(1-batch['done'])*min_q_targ

            q1_loss = tf.reduce_mean((q1 - backup)**2)
            q2_loss = tf.reduce_mean((q2 - backup)**2)
            q_loss = q1_loss + q2_loss

            if debug_mode and self.cur_tensorboard_step % 1 == 0:
                tensorboard_step = int(self.cur_tensorboard_step/1)
                with self.summary.as_default():
                    tf.summary.scalar("loss_q1", q1_loss, tensorboard_step)
                    tf.summary.scalar("loss_q2", q2_loss, tensorboard_step)
                    tf.summary.scalar("loss_q", q_loss, tensorboard_step)
                    tf.summary.histogram("q1", q1, tensorboard_step)
                    tf.summary.histogram("q2", q2, tensorboard_step)
                    tf.summary.histogram("pi_targ", pi_targ, tensorboard_step)
                    tf.summary.histogram("pi_a2", a2, tensorboard_step)

        train_vars = self.actor_critic.q1.trainable_variables + \
               self.actor_critic.q2.trainable_variables + \
               self.actor_critic.actor.emd.trainable_variables
        # train_vars = self.actor_critic.actor.emd.trainable_variables

        q_gradient = tape.gradient(q_loss, train_vars)
        self.opti_q.apply_gradients(zip(q_gradient, train_vars))

    def train_p(self, batch):

        with tf.GradientTape() as tape:
            img1 = batch['obs1']["img"]
            speed1 = batch['obs1']["speed"]
            direction1 = batch['obs1']["direction"]
            direction1 = tf.stack([tf.range(self.batch_size), direction1], axis=1)    # [None, 2]

            pi_list, z = self.actor_critic.select_action([img1, speed1])
            pi_list = tf.stack(pi_list[0:4], axis=1)  # [None, 4, 3]
            pi = tf.gather_nd(pi_list, direction1)   # [None, 3]

            q1_pi_list = self.actor_critic.work_q1(z, pi)
            # q1_pi_list, _ = self.actor_critic([img1, speed1, pi])
            q1_pi_list = tf.stack(q1_pi_list, axis=1)
            q1_pi = tf.gather_nd(q1_pi_list, direction1)

            pi_loss = -tf.reduce_mean(q1_pi)

        train_vars_pi = self.actor_critic.actor.trainable_variables
        pi_gradient = tape.gradient(pi_loss, train_vars_pi)
        train_pi = [var for (var, gra) in zip(train_vars_pi, pi_gradient) if gra is not None]
        pi_gra = [gra for (var, gra) in zip(train_vars_pi, pi_gradient) if gra is not None]
        self.opti_pi.apply_gradients(zip(pi_gra, train_pi))
        self.target_update(self.target_actor_critic, self.actor_critic)

        if debug_mode and self.cur_tensorboard_step % 1 == 0:
            tensorboard_step = int(self.cur_tensorboard_step/1)
            with self.summary.as_default():
                tf.summary.histogram("pi", pi, tensorboard_step)
                tf.summary.histogram("q1_pi", q1_pi, tensorboard_step)
                tf.summary.scalar("loss_pi", pi_loss, tensorboard_step)

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

                if self.cur_train_step > self.start_steps or eval_mode or True:
                    a = self.get_action(obs, self.act_noise, eval_mode)

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
                    self.replay_buffer.add(obs, a, obs_, [r], [done])

                if step_episode >= self.max_ep_len:
                    break
                obs = obs_

            episode += 1
            if self.cur_train_step > self.start_steps and not eval_mode:
                for j in range(step_episode):
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.cur_tensorboard_step += 1

                    self.train_q(batch)

                    if j % self.policy_delay == 0:
                        self.train_p(batch)

            if episode % 20 == 0 and not eval_mode:
                self.manager.save()

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
            # self.logger.log_tabular('Q1Vals', with_min_and_max=True)
            # self.logger.log_tabular('Q2Vals', with_min_and_max=True)
            # self.logger.log_tabular('LossPi', average_only=True)
            # self.logger.log_tabular('LossQ', average_only=True)
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
    parser.add_argument('--exp_name', type=str, default='td3_carla_test')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iteration', type=int, default=2)
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    td3 = Td3(args.env, port=args.port, gpu=args.gpu, iteration=args.iteration, logger_kwargs=logger_kwargs)
    td3.train_test()
