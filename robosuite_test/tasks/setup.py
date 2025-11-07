# setup.py
from setuptools import setup, find_packages

setup(
    name='multi_task_robosuite_env',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['test_models']),
    package_data={"multi_task_robosuite_env.objects": ['*','*/*','*/*/*','*/*/*/*'],
                  "multi_task_robosuite_env.arena": ['*','*/*'],
                  "multi_task_robosuite_env.controllers": ['*','*/*'],
                  "multi_task_robosuite_env.utils": ['*'],
                  "multi_task_robosuite_env.config": ['*']}
)

if __name__ == "__main__":
    packages = find_packages(exclude=['test_models'])
    print(packages)
