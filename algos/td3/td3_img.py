import tensorflow as tf
from algos.td3 import core
from utils.run_utils import setup_logger_kwargs
import argparse
import os.path as osp
from env.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3_carla')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    filepath = osp.join(logger_kwargs["output_dir"], "saver")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session("", config=config)
    sess.run(tf.global_variables_initializer())

    actor_critic = core.ActorCritic(act_dim=2)
    actor_critic.load_weights(osp.join(filepath, "model_30"))
    # check = tf.train.Checkpoint(model=actor_critic)
    # check.restore(tf.train.latest_checkpoint(filepath))

    # conv1 = actor_critic.get_layer("conv1").output
    # conv2 = actor_critic.get_layer("conv2").output
    # conv3 = actor_critic.get_layer("conv3").output
    #
    # summary = tf.summary.FileWriter("log")
    # tf.summary.image("conv1", conv1)
    # tf.summary.image("conv2", conv2)
    # tf.summary.image("conv3", conv3)
    env = CarlaEnv()
    s_ph = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 6])

    s = env.reset()
    done = False
    while True:
        s = np.array(s).astype(np.float32)/255.0
        s = np.expand_dims(s, axis=0)
        a = actor_critic.select_action(s_ph)
        act = sess.run(a, feed_dict={s_ph: s})
        print(act)

        s, r, d, _ = env.step(act)






