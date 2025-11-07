from setuptools import setup, find_packages

setup(
    name='Visual One-shot Imitation',
    version='0.0.1',
    packages=find_packages(),
)

if __name__ == "__main__":
    packages = find_packages()
    print(packages)