import tensorflow as tf
import algos.gail.core as core
import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import os


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=10)
    parser.add_argument('--d_update', type=int, default=5)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--env', type=str, default="CartPole-v0")
    args = parser.parse_args()

    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    env = gym.make(args.env)
    ppo = core.PPO(env.action_space.n, 0.2)
    discriminator = core.Discriminator()
    opti_d = tf.keras.optimizers.Adam(0.001)
    summary = tf.summary.create_file_writer(os.path.join(args.log, args.env))

    for iter in range(args.iteration):
        observations = []
        actions = []
        vpreds = []
        rew = 0

        obs = env.reset()

        while True:
            a = ppo.actor.select_action(obs[np.newaxis, :])
            obs_, r, done, _ = env.step(a.numpy()[0])
            rew += r
            v_pred = tf.squeeze(ppo.get_v(obs[np.newaxis, :]), axis=-1)

            observations.append(obs)
            actions.append(a)
            vpreds.append(v_pred)

            if done:
                v_next = tf.squeeze(ppo.get_v(obs_[np.newaxis, :]), axis=-1)
                vpreds_next = vpreds[1:] + [v_next]

                break

            obs = obs_

        with summary.as_default():
            tf.summary.scalar("reard", rew, iter)
        print("iter:", iter, "rewards:", rew)

        batch_size = len(observations)
        agent_s = np.stack(observations)
        agent_a = np.stack(actions)
        agent = [agent_s, agent_a]
        for i in range(args.d_update):
            sample_indices = np.random.choice(range(np.shape(expert_observations)[0]), size=batch_size)
            expert_s = expert_observations[sample_indices]
            expert_a = expert_actions[sample_indices]
            expert_a = expert_a[:, np.newaxis]
            expert = [expert_s, expert_a]

            with tf.GradientTape() as tape:
                expert_d = discriminator(expert)
                agent_d = discriminator(agent)

                loss_expert = -tf.reduce_mean(tf.math.log(expert_d))
                loss_agent = -tf.reduce_mean(tf.math.log(1-agent_d))
                loss = loss_agent + loss_expert

            grad = tape.gradient(loss, discriminator.trainable_variables)
            opti_d.apply_gradients(zip(grad, discriminator.trainable_variables))

        with summary.as_default():
            tf.summary.scalar("loss_d", loss, iter)

        rewards = tf.math.log(agent_d)
        vpreds = tf.stack(vpreds, axis=0)
        vpreds_next = tf.stack(vpreds_next, axis=0)
        adv = rewards + args.gamma*vpreds_next - vpreds
        adv = discount_cumsum(adv, args.gamma*args.lam)
        vs = rewards + args.gamma*vpreds_next
        vs = discount_cumsum(rewards, args.gamma)

        ppo.update_a()
        for i in range(args.a_update):
            aloss = ppo.train_a(agent_s, agent_a, adv)

        for i in range(args.c_update):
            vloss = ppo.train_v(agent_s, vs)

        with summary.as_default():
            tf.summary.scalar("loss_a", aloss, iter)
            tf.summary.scalar("loss_v", vloss, iter)










