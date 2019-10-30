from carla.agent.forward_agent import ForwardAgent
from carla.driving_benchmark.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import basic_experiment_suite
import subprocess
import argparse

def start_carla_simulator(port=2000, gpu=0, town_name="Town01", docker="carlasim/carla:0.8.4"):



    # sp = subprocess.Popen(['docker', 'run', '--rm', '-d', '-p',
    #                        str(port)+'-'+str(port+2)+':'+str(port)+'-'+str(port+2),
    #                        '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES='+str(gpu), docker,
    #                        '/bin/bash', 'CarlaUE4.sh', '/Game/Maps/' + town_name, '-windowed',
    #                        '-benchmark', '-fps=10', '-world-port=' + str(port)], shell=False,
    #                        stdout=subprocess.PIPE)

    sp = subprocess.Popen(['docker', 'run', '--rm', '-d', '-p',
                           str(port)+'-'+str(port+2)+':'+str(port)+'-'+str(port+2),
                           '--gpus='+str(gpu), docker,
                           '/bin/bash', 'CarlaUE4.sh', '/Game/Maps/' + town_name, '-windowed',
                           '-benchmark', '-fps=10', '-world-port=' + str(port)], shell=False,
                           stdout=subprocess.PIPE)

    (out, err) = sp.communicate()
    print("Going to communicate")

    return sp, out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--docker', type=str, default="carlasim/carla:0.8.4")
    args = parser.parse_args()

    start_carla_simulator(port=args.port, gpu=args.gpu, town_name=args.town, docker=args.docker)
    agent = ForwardAgent()
    experiment = basic_experiment_suite.BasicExperimentSuite(args.town)

    run_driving_benchmark(agent, experiment, city_name=args.town, host=args.host, port=args.port)

