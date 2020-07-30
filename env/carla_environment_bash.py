from user_config import DEFAULT_CARLA_LOG_DIR

import sys, time, subprocess, signal
from os import path, environ

try:
    if 'CARLA_ROOT' in environ:
        sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
except ImportError:
    print("CARLA Environment variable CARLA_ROOT not set")
    sys.exit(1)
from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.sensor import Camera
from carla.client import VehicleControl
from carla.image_converter import depth_to_logarithmic_grayscale, depth_to_local_point_cloud, depth_to_array
from env.renderer import Renderer

import numpy as np
from env.environment_wrapper import EnvironmentWrapper
from env.utils import *
import env.carla_config as carla_config
from carla.driving_benchmark.experiment_suites import CoRL2017
from carla.planner.planner import Planner

import cv2
from gym.spaces import Box, Discrete, Tuple
import time
import random
import math
from subprocess import getoutput as shell

# enum of the available levels and their path
class CarlaLevel(Enum):
    TOWN1 = "/Game/Maps/Town01"
    TOWN2 = "/Game/Maps/Town02"

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

class CarlaEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, num_speedup_steps=30, require_explicit_reset=True, is_render_enabled=False,
                 early_termination_enabled=False, run_offscreen=False, save_screens=False,
                 port=2000, gpu=0, discrete_control=True, kill_when_connection_lost=True, city_name="Town01",
                 channel_last=True, action_num=2):
        EnvironmentWrapper.__init__(self, is_render_enabled, save_screens)

        print("port:", port)

        self.episode_max_time = 1000000
        self.allow_braking = True
        self.log_path = os.path.join(DEFAULT_CARLA_LOG_DIR, "CarlaLogs.txt")
        self.num_speedup_steps = num_speedup_steps
        self.is_game_ready_for_input = False
        self.run_offscreen = run_offscreen
        self.kill_when_connection_lost = kill_when_connection_lost
        # server configuration


        self.port = port
        self.gpu = gpu
        self.host = 'localhost'
        self.level = 'town1'
        self.map = CarlaLevel().get(self.level)

        # experiment = basic_experiment_suite.BasicExperimentSuite(city_name)
        experiment = CoRL2017(city_name)
        self.experiments = experiment.get_experiments()
        self.experiment_type = 0
        self.planner = Planner(city_name)

        self.car_speed = 0
        self.is_game_setup = False  # Will be true only when setup_client_and_server() is called, either explicitly, or by reset()

        # action space
        self.discrete_controls = discrete_control
        self.action_space_size = action_num
        self.action_space_high = np.array([1]*action_num)
        self.action_space_low = np.array([-1]*action_num)
        self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
        self.steering_strength = 0.35
        self.gas_strength = 1.0
        self.brake_strength = 0.6
        self.actions = {0: [0., 0.],
                        1: [0., -self.steering_strength],
                        2: [0., self.steering_strength],
                        3: [self.gas_strength - 0.15, 0.],
                        4: [-self.brake_strength, 0],
                        5: [self.gas_strength - 0.3, -self.steering_strength],
                        6: [self.gas_strength - 0.3, self.steering_strength],
                        7: [-self.brake_strength, -self.steering_strength],
                        8: [-self.brake_strength, self.steering_strength]}
        self.actions_description = ['NO-OP', 'TURN_LEFT', 'TURN_RIGHT', 'GAS', 'BRAKE',
                                    'GAS_AND_TURN_LEFT', 'GAS_AND_TURN_RIGHT',
                                    'BRAKE_AND_TURN_LEFT', 'BRAKE_AND_TURN_RIGHT']
        if discrete_control:
            self.action_space = Discrete(len(self.actions))
        else:
            self.action_space = Box(low=self.action_space_low, high=self.action_space_high)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=[88, 200, 3])

        # measurements
        self.measurements_size = (1,)

        self.pre_image = None
        self.first_debug = True
        self.channel_last = channel_last

    def setup_client_and_server(self, reconnect_client_only=False):
        # open the server
        if not reconnect_client_only:
            self.server = self._open_server()
            self.server_pid = self.server.pid  # To kill incase child process gets lost
            print("setup server, out:", self.server_pid)

        while True:
            try:
                self.game = CarlaClient(self.host, self.port, timeout=99999999)
                self.game.connect(
                    connection_attempts=100)  # It's taking a very long time for the server process to spawn, so the client needs to wait or try sufficient no. of times lol
                self.game.load_settings(CarlaSettings())
                self.game.start_episode(0)

                self.is_game_setup = self.server and self.game

                print("setup client")

                return
            except TCPConnectionError as error:
                print(error)
                time.sleep(1)

    def close_client_and_server(self):
        self._close_server()
        print("Disconnecting the client")
        self.game.disconnect()
        self.game = None
        self.server = None
        self.is_game_setup = False
        return

    def _open_server(self):
        # Note: There is no way to disable rendering in CARLA as of now
        # https://github.com/carla-simulator/carla/issues/286
        # decrease the window resolution if you want to see if performance increases
        # Command: $CARLA_ROOT/CarlaUE4.sh /Game/Maps/Town02 -benchmark -carla-server -fps=15 -world-port=9876 -windowed -ResX=480 -ResY=360 -carla-no-hud
        # To run off_screen: SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 <command> #https://github.com/carla-simulator/carla/issues/225
        my_env = None
        if self.run_offscreen:
            my_env = {**os.environ, 'SDL_VIDEODRIVER': 'offscreen', 'SDL_HINT_CUDA_DEVICE': '0', }
        with open(self.log_path, "wb") as out:

            cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map,
                   "-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(self.port),
                   "-windowed -ResX={} -ResY={}".format(carla_config.server_width, carla_config.server_height),
                   "-carla-no-hud"]
            p = subprocess.Popen(cmd, stdout=out, stderr=out, env=my_env, preexec_fn=os.setsid)

        return p

    def _close_server(self):
        if self.kill_when_connection_lost:
            print("kill before")
            os.kill(os.getpgid(self.server.pid), signal.SIGKILL)
            # self.server.kill()

            no_of_attempts = 0
            while is_process_alive(self.server_pid):
                print("Trying to close Carla server with pid %d" % self.server_pid)
                if no_of_attempts < 5:
                    self.server.terminate()
                elif no_of_attempts < 10:
                    self.server.kill()
                elif no_of_attempts < 15:
                    os.kill(self.server_pid, signal.SIGTERM)
                else:
                    os.kill(self.server_pid, signal.SIGKILL)
                time.sleep(10)
                no_of_attempts += 1

            print("kill after")



    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self.planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _update_state(self):
        # get measurements and observations
        measurements = []
        while type(measurements) == list:
            try:
                measurements, sensor_data = self.game.read_data()

            except:

                print("Connection to server lost while reading state. Reconnecting...........")
                # self.game.disconnect()
                # self.setup_client_and_server(reconnect_client_only=True)
                import psutil
                info = psutil.virtual_memory()
                print("memory used", str(info.percent))

                self.close_client_and_server()
                self.setup_client_and_server(reconnect_client_only=False)

                scene = self.game.load_settings(self.cur_experiment.conditions)
                self.positions = scene.player_start_spots
                self.start_point = self.positions[self.pose[0]]
                self.end_point = self.positions[self.pose[1]]
                self.game.start_episode(self.pose[0])

                self.done = True

        current_point = measurements.player_measurements.transform
        direction = self._get_directions(current_point, self.end_point)
        speed = measurements.player_measurements.forward_speed

        self.reward = 0
        dist = sldist([current_point.location.x, current_point.location.y], [self.end_point.location.x, self.end_point.location.y])

        if direction == 5:               #go straight
            if abs(self.control.steer)>0.2:
                self.reward -= 20

            self.reward += min(35, speed*3.6)
        elif direction == 2:             #follow lane
            self.reward += min(35, speed*3.6)
        elif direction == 3:             #turn left ,steer negtive
            if self.control.steer > 0:
                self.reward -= 15
            if speed*3.6 <= 20:
                self.reward += speed*3.6
            else:
                self.reward += 40 - speed*3.6
        elif direction == 4:            #turn right
            if self.control.steer < 0:
                self.reward -= 15
            if speed*3.6 <= 20:
                self.reward += speed*3.6
            else:
                self.reward += 40 - speed*3.6
        elif direction == 0:
            self.done = True
            self.reward += 100
            direction = 2
            print("success", dist)
        else:
            print("error direction")
            direction = 5

        direction -= 2
        direction = int(direction)

        intersection_offroad = measurements.player_measurements.intersection_offroad
        intersection_otherlane = measurements.player_measurements.intersection_otherlane
        collision_veh = measurements.player_measurements.collision_vehicles
        collision_ped = measurements.player_measurements.collision_pedestrians
        collision_other = measurements.player_measurements.collision_other

        if intersection_otherlane > 0 or intersection_offroad > 0 or collision_ped > 0 or collision_veh > 0:
            self.reward -= 100
        elif collision_other > 0:
            self.reward -= 50

        if collision_other > 0 or collision_veh > 0 or collision_ped > 0:
            self.done = True
        if intersection_offroad > 0.2 or intersection_otherlane > 0.2:
            self.done = True
        if speed*3.6 <= 1:
            self.nospeed_time += 1
            if self.nospeed_time > 100:
                self.done = True
            self.reward -= 1
        else:
            self.nospeed_time = 0

        # speed = min(1, speed/10.0)
        speed = speed/25

        # print(measurements)
        # print(sensor_data)
        self.observation = {
            "img": self.process_image(sensor_data['CameraRGB'].data),
            "speed": speed,
            "direction": direction
        }
        return self.observation, self.reward, self.done

    def encode_obs(self, image):
        if self.pre_image is None:
            self.pre_image = image

        img = np.concatenate([self.pre_image, image], axis=2)
        self.pre_image = image
        return img

    def process_image(self, data):
        rgb_img = data[115:510, :]
        rgb_img = cv2.resize(rgb_img, (200, 88), interpolation=cv2.INTER_AREA)
        if not self.channel_last:
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # return cv2.resize(
        #     data, (80, 80),
        #     interpolation=cv2.INTER_AREA)
        return rgb_img

    def _take_action(self, action_idx):
        if not self.is_game_setup:
            print("Reset the environment duh by reset() before calling step()")
            sys.exit(1)
        if type(action_idx) == np.int64 or type(action_idx) == np.int:
            action = self.actions[action_idx]
        else:
            action = action_idx

        self.control = VehicleControl()

        if len(action) == 3:
            self.control.throttle = np.clip(action[1], 0, 1)
            self.control.steer = np.clip(action[0], -1, 1)
            self.control.brake = np.clip(action[2], 0, 1)
        else:
            self.control.throttle = np.clip(action[0], 0, 1)
            self.control.steer = np.clip(action[1], -1, 1)
            self.control.brake = np.abs(np.clip(action[0], -1, 0))

        if not self.allow_braking:
            self.control.brake = 0
        self.control.hand_brake = False
        self.control.reverse = False
        controls_sent = False
        while not controls_sent:
            try:
                self.game.send_control(self.control)
                controls_sent = True
                # print(self.control)
                # #
                # rand = random.randint(0, 7)
                # if rand == 0 and self.first_debug:
                #     self.first_debug = False
                #     raise Exception
            except:
                print("Connection to server lost while sending controls. Reconnecting...........")
                import psutil
                info = psutil.virtual_memory()
                print("memory used", str(info.percent))

                self.close_client_and_server()
                self.setup_client_and_server(reconnect_client_only=False)

                scene = self.game.load_settings(self.cur_experiment.conditions)
                self.positions = scene.player_start_spots
                self.start_point = self.positions[self.pose[0]]
                self.end_point = self.positions[self.pose[1]]
                self.game.start_episode(self.pose[0])

                self.done = True
                controls_sent = False
        return

    def _restart_environment_episode(self, force_environment_reset=True):

        if not force_environment_reset and not self.done and self.is_game_setup:
            print("Can't reset dude, episode ain't over yet")
            return None  # User should handle this
        self.is_game_ready_for_input = False
        if not self.is_game_setup:
            self.setup_client_and_server()

        experiment_type = random.randint(0, 2)
        self.cur_experiment = self.experiments[experiment_type]
        self.pose = random.choice(self.cur_experiment.poses)
        scene = self.game.load_settings(self.cur_experiment.conditions)
        self.positions = scene.player_start_spots
        self.start_point = self.positions[self.pose[0]]
        self.end_point = self.positions[self.pose[1]]

        while True:
            try:
                self.game.start_episode(self.pose[0])

                # start the game with some initial speed
                self.car_speed = 0
                self.nospeed_time = 0
                observation = None
                for i in range(self.num_speedup_steps):
                    observation, reward, done, _ = self.step([0, 1.0, 0])
                self.observation = observation
                self.is_game_ready_for_input = True
                break

            except Exception as error:
                print(error)
                self.game.connect()
                self.game.start_episode(self.pose[0])

        return observation


if __name__ == '__main__':

    env = CarlaEnvironmentWrapper()
    try:
        for _ in range(5):
            # env = CarlaEnv()
            obs = env.reset(force_environment_reset=False)
            done = False
            t = 0
            total_reward = 0.0
            while not done:
                t += 1

                # file = os.path.join("imgs", "step"+str(t)+".png")
                # cv2.imwrite(file, obs["img"])
                obs, reward, done, info = env.step(3)  # Go Forward

                total_reward += reward
                print("step#:", t, "reward:", round(reward, 4), "total_reward:", round(total_reward, 4), "done:", done)

    except Exception as error:
        print(error)
        env.close_client_and_server()
