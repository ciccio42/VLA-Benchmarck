import yaml, os, sys
from yaml import Loader, Dumper
import numpy as np

def read_conf_file(task_name: str) -> dict:

    # path to the current directory
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # create path to configuration file
    conf_file_path = os.path.join(dir_path, "../config", f"{task_name}.yaml")
    
    with open(conf_file_path, "r") as conf_file:
        data = yaml.load(conf_file, Loader=Loader)

    return data

if __name__ == '__main__':
    print("Testing read_conf_file")
    env_conf = read_conf_file("pick_place")
    print(env_conf)