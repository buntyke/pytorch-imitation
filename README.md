# pytorch-imitation

This repository contains an implementation of imitation learning algorithms for different OpenAI gym environments. The implementation is modular and can easily be extended to novel environments, policies and loss functions.

## Getting Started

These instructions will get you started with the repository. 


### Prerequisites

* This package is supported for `python 3.6`. Setup the python environment using [Anaconda](https://www.anaconda.com/download)/[Miniconda](https://conda.io/miniconda.html) (preferred).


* The default environment supported by this package is the OpenAI gym mujoco environments. For this you will need to install Mujoco library. Follow instructions from [here](https://github.com/openai/mujoco-py#install-mujoco).

  Add the following line to `~/.bashrc` file: 
  ```
  LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
  ```

* Install dependencies for `mujoco-py`, python wrapper for Mujoco. Current support for Linux (Debian): 
  ```
  $ sudo apt install curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra
  ```

* (Optional) Install NVidia driver and CUDA 9.0 toolkit for GPU support. Follow instructions for Linux(Debian) from [here](https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/).

  Add the following line to `.bashrc` file:
  ```
  LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
  ```

### Installation

* Create a new conda environment to avoid conflicts:
  ```
  $ conda create -n imitation python=3.6 
  ```

* Install the requirements to run the library on activating the environment:
  ```
  $ conda activate imitation
  ```

  ```
  (imitation) $ pip install numpy matplotlib tqdm
  (imitation) $ pip install mujoco-py gym gym[mujoco]
  (imitation) $ pip install torch torchvision tensorboardX
  ```

### Usage

* Every experiment requires a `json` file to provide the configuration. A config file for the `Ant-v2` environment is provided.

* Download the dataset for `Ant-v2` environment from [here](https://github.com/buntyke/pytorch-imitation/releases/download/v1.0/Ant-v2.pkl) and place it in the following folder:
  ```
  pytorch-imitation/data/Mujoco/Ant-v2.pkl
  ```

* Train the policy network using the ant config file:
  ```
  (imitation) $ python train.py -c ant.json
  ```

* Model performance can be viewed by using tensorboard:
  ```
  (imitation) $ cd saved/runs/Ant/1217_172542/
  (imitation) $ tensorboard --logdir=.
  ```

  The model training curves can be visualized at this URL: `https://localhost:6006`

* Test the policy network using the `test.py` script:
  ```
  (imitation) $ python test.py -m saved/Ant/1217_172542/model_best.pth -e Ant-v2
  ```
  The path to the model changes depending on the timestamp.

### Additional

* Behavioral cloning requires expert demonstrations to train the policy network. Generate the expert demonstrations using the Berkeley Deep RL course assignment [here](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw1). Demonstrations for `Ant-v2` environment provided along with the package.

* To generalize implementation to other environments the following components can be changed `trainer`, `model`, `data_loader`.

### Authors

* [Nishanth Koganti](buntyke.github.io)

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

### Acknowledgements

* README template obtained from this [gist file](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2).
* Pytorch project template was obtained from this [repository](https://github.com/victoresque/pytorch-template).
* Expert policies for mujoco-envs were obtained from Berkeley [DRL course](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw1).
