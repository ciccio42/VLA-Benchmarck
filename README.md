Usefull [link](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)


# Robosuite for OpenVLA Installation
In `robosuite_test` folder:
```bash
conda env create -f conda_environments/openvla_robosuite_1_0_1.yaml
# Install OpenVLA
pip install -r python_requirements/openvla_requiments.txt  # openvla-requirements
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git
cd tasks/training
pip install -e .
pip install pyquaternion

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
cd ..
source install.sh
```

# Robosuite for TinyVLA Installation
In `robosuite_test` folder:
```bash
conda env create -f conda_environments/tinyvla_robosuite_1_0_1.yaml
pip install -r python_requirements/tinyvla_requirements.txt
pip install mergedeep~=1.3 pyyaml-include~=1.4 toml~=0.10 typing-inspect~=0.9.0 wrapt tensorflow>=2.2.0 protobuf>=3.20
# Install torch 
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 –index-url https://download.pytorch.org/whl/cu128
# Install utils
pip install -e ../.

git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt 
cd ..
source install.sh
```

Install TinyVLA modules

```bash
cd [PATH-TO-TinyVLA-Folder]
pip install -e .
cd policy_heads
pip install -e .
# install llava-pythia
cd ../llava-pythia
pip install -e . 
```