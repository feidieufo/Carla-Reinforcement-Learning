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
from carla.carla_server_pb2 import Control
import env.carla_config as carla_config
import cv2
import scipy
import os
import torch
from algos.imitation.carla_net import CarlaNet

from carla.driving_benchmark.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import basic_experiment_suite, corl_2017

class ImitAgent(Agent):
    def __init__(self, model):
        self.model = model
        self.image_size = (88, 200, 3)
        self.image_cut = [115, 510]
        self.step = 0

    def run_step(self, measurements, sensor_data, directions, target):
        img = sensor_data['CameraRGB'].data

        filepath = os.path.join(DEFAULT_DATA_DIR, "imgs")
        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + ".png"), img)

        rgb_image = img[self.image_cut[0]:self.image_cut[1], :]
        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "xx.png"), rgb_image)



        image_input = cv2.resize(
            rgb_image, (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_AREA)

        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "yy.png"), image_input)

        self.step += 1

        # image_input = scipy.misc.imresize(rgb_image, [self.image_size[0],
        #                                               self.image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        speed = measurements.player_measurements.forward_speed/25
        # speed = 0

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


class ImitAgent_torch(Agent):
    def __init__(self, model):
        self.model = model
        self.image_size = (88, 200, 3)
        self.image_cut = [115, 510]
        self.step = 0

    def run_step(self, measurements, sensor_data, directions, target):
        img = sensor_data['CameraRGB'].data

        filepath = os.path.join(DEFAULT_DATA_DIR, "imgs")
        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + ".png"), img)

        rgb_image = img[self.image_cut[0]:self.image_cut[1], :]
        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "xx.png"), rgb_image)



        image_input = cv2.resize(
            rgb_image, (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_AREA)

        # cv2.imwrite(os.path.join(filepath, "img_" + str(self.step) + "yy.png"), image_input)

        self.step += 1

        # image_input = scipy.misc.imresize(rgb_image, [self.image_size[0],
        #                                               self.image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        speed = measurements.player_measurements.forward_speed/25
        # speed = 0

        image_input = np.expand_dims(image_input, 0)
        speed = np.expand_dims(speed, 0)
        speed = np.expand_dims(speed, 0)

        self.model.eval()
        preds = self.model(image_input, speed)
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
    parser.add_argument('--saver', type=str, default='saver_0.45_0.45_0.45_0.3')
    parser.add_argument('--filename', type=str, default='model.ckpt-199')

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    filepath = os.path.join(DEFAULT_DATA_DIR, args.saver)
    model = core.ActorCnn()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(filepath))
    checkpoint.restore(os.path.join(filepath, args.filename))

    # filepath = os.path.join(os.path.join("data", "torch_1_0.1"), "checkpoint_2.pth")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = CarlaNet()
    # model.to(device)
    # model.load_state_dict(torch.load(filepath)['state_dict'])
    # agent = ImitAgent_torch(model)



    agent = ImitAgent(model)
    start_carla_simulator(port=args.port, gpu=args.gpu, town_name=args.town)
    # experiment = basic_experiment_suite.BasicExperimentSuite(args.town)
    experiment = corl_2017.CoRL2017(args.town)

    run_driving_benchmark(agent, experiment, city_name=args.town, host=args.host, port=args.port, log_name=args.saver)




