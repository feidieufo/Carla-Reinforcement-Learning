import tensorflow as tf
from user_config import DEFAULT_DATA_DIR
import os
import argparse
from os import path, environ
import numpy as np
import subprocess
import algos.imitation.core as core
from env.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from carla.agent.agent import Agent
from algos.imitation_baseline.imitation_learning import ImitationLearning
from carla.carla_server_pb2 import Control
import env.carla_config as carla_config
import cv2
import scipy
import os

from carla.driving_benchmark.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import basic_experiment_suite
from algos.imitation_baseline.imitation_learning_network import load_imitation_learning_network


class ImitAgent(Agent):
    def __init__(self, city_name, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):
        Agent.__init__()
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        config_gpu = tf.ConfigProto()

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        config_gpu.gpu_options.visible_device_list = '0'

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        self._sess = tf.Session(config=config_gpu)

        with tf.device('/gpu:0'):
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                self._image_size[1],
                                                                self._image_size[2]],
                                                name="input_image")

            self._input_data = []

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4], name="input_control"))

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1], name="input_speed"))

            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                                   self._input_data,
                                                                   self._image_size, self._dout)
        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        # tf.reset_default_graph()
        self._sess.run(tf.global_variables_initializer())

        variables_to_restore = tf.global_variables()

        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        self._image_cut = image_cut
        self.step = 0

    def run_step(self, measurements, sensor_data, directions, target):
        img = sensor_data['CameraRGB'].data

        filepath = os.path.join(DEFAULT_DATA_DIR, "imgs")
        cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + ".png"), img)

        rgb_image = img[self._image_cut[0]:self._image_cut[1], :]
        cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "xx.png"), rgb_image)

        image_input = cv2.resize(
            rgb_image, (self._image_size[1], self._image_size[0]),
            interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "yy.png"), image_input)

        self.step += 1

        # image_input = scipy.misc.imresize(rgb_image, [self.image_size[0],
        #                                               self.image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        # speed = measurements.player_measurements.forward_speed
        speed = 0

        image_input = np.expand_dims(image_input, 0)
        speed = np.expand_dims(speed, 0)
        speed = np.expand_dims(speed, 0)

        preds = self.model([image_input, speed], False)
        if directions == 0.0:
            pred = preds[0]
        else:
            directions -= 2
            directions = int(directions)
            pred = preds[directions]

        steer = pred[0][0]
        acc = pred[0][1]
        brake = pred[0][2]

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        print("steer:", control.steer, "throttle:", control.throttle, "brake:", control.brake)

        return control


def start_carla_simulator(port=2000, gpu=0, town_name="Town01", run_offscreen=False):
    my_env = None
    if run_offscreen:
        my_env = {**os.environ, 'SDL_VIDEODRIVER': 'offscreen', 'SDL_HINT_CUDA_DEVICE': '0', }

    cmd = [os.path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), '/Game/Maps/' + town_name,
           "-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(port),
           "-windowed -ResX={} -ResY={}".format(carla_config.server_width, carla_config.server_height),
           "-carla-no-hud"]

    p = subprocess.Popen(cmd, env=my_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--town', type=str, default='Town02')
    parser.add_argument('--saver', type=str, default='saver_0.45_0.45_0.05_0.1')
    parser.add_argument('--file', type=str, default='model.ckpt-9')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # filepath = os.path.join(DEFAULT_DATA_DIR, args.saver)
    # model = core.ActorCnn()
    # checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(filepath))
    # checkpoint.restore(os.path.join(filepath, args.file))

    agent = ImitationLearning(args.town, True)
    start_carla_simulator(port=args.port, gpu=args.gpu, town_name=args.town)
    experiment = basic_experiment_suite.BasicExperimentSuite(args.town)

    run_driving_benchmark(agent, experiment, city_name=args.town, host=args.host, port=args.port)



