# Robotics Control Theory

## Project Description

The goal of this project is to implement and analyze an adaptive algorithm inspired by the RMA (Rapid Motor Adaptation) approach for a legged robot in the Genesis simulation environment. The model is designed to generate stable robot locomotion under changing environmental conditions, such as different friction values, slippery surfaces, and uneven terrain.

As part of the project, two components were trained: a locomotion control policy and an adaptation module that estimates the hidden state of the environment based on the history of robot states and actions.

---

## Goals

- Implementation of a locomotion control policy
- Implementation of an adaptation module
- Model training in the Genesis environment
- Comparison of the adaptive model performance under different test conditions

---

## Environment Setup

The project was prepared and tested on Linux. Python is required to run it, as well as access to the `genesis` and `rsl_rl` submodules, because the project code relies on specific versions of these libraries.

First, clone the repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/krukich/TSwR.git
cd TSwR
```

If the repository has already been cloned without submodules, initialize them separately:

```bash
git submodule update --init --recursive
```

Then run the installation script:

```bash
bash scripts/setup.sh
```

The script creates a `.venv` virtual environment, installs the required dependencies, and checks the availability of the submodules and Unitree Go2 robot files.

After installation, activate the environment with:

```bash
source .venv/bin/activate
```

From this point, the training and evaluation scripts can be executed from the project directory. The trained models are not stored directly in the repository, so they must be downloaded separately from Google Drive.

Ready-to-use commands for running individual experiments are included in the descriptions of the model files on Google Drive.

## Trained Models

The trained models are not stored directly in the repository because of the checkpoint file sizes. All current models are available in a separate Google Drive folder:

[Google Drive - trained models](https://drive.google.com/drive/folders/1x3uDen8MqZw5EVCSddOEvBN7ynM8qbgJ?usp=drive_link)

Each model directory contains a `description.txt` file together with the checkpoint files. This file includes a short description of the model, training configuration, and the recommended way to use it.

---

## Architecture

The project consists of two main components:

- **Base Policy (π)** – generates actions based on the current state of the robot
- **Adaptation Module (φ)** – estimates the latent state \(z_t\) based on the history of states and actions

---

## File Description

### Main Project Files

- `src/go2_env.py` – the main simulation environment file for the Unitree Go2 robot. It defines the environment dynamics, observations, actions, rewards, episode resets, and randomization of parameters such as friction and additional payload mass.

- `src/go2_train.py` – the script used to start training the locomotion control policy in the `Go2Env` environment. It contains the PPO configuration, environment parameters, command ranges, and checkpoint saving settings.

- `src/go2_eval.py` – a simple evaluation script used to run a trained policy in a single environment with visualization. It allows checking the model behavior for a selected checkpoint, target velocity, and base height.

- `src/go2_eval_friction.py` – an extended testing script used to evaluate the policy under different friction and payload conditions. It collects basic metrics such as average velocity, number of falls, episode length, and reset reasons.

- `src/check.py` – a helper diagnostic script that verifies correct loading of the Go2 model and the mapping of the 12 controlled joints. It is mainly used for quick verification of the URDF configuration and DOF indices.

### Configuration and Running

- `scripts/setup.sh` – the project setup script. It creates the `.venv` environment, installs dependencies, initializes the `genesis` and `rsl_rl` submodules, and checks the availability of the Go2 robot URDF files.

### External Directories

The `genesis/` and `rsl_rl/` directories are used in this project as submodules, because the environment and training code were prepared for specific versions of the Genesis and rsl_rl libraries.

- `genesis/` – a submodule containing the Genesis simulator used to create the physical simulation environment.
- `rsl_rl/` – a submodule containing the reinforcement learning library used for PPO policy training.

## Experiment Results

As part of the project, experiments were conducted on training a locomotion control policy for the Unitree Go2 robot in the Genesis simulator and testing its robustness under changing environmental conditions. The main policy was trained using PPO with the `rsl_rl` library, and the robot was controlled in position mode for 12 joints.

The first stage was to train a base policy that received the robot's proprioceptive observations and information about the commanded movement velocity. After training, the robot learned a stable forward gait and was able to maintain balance under standard environment settings. Then, generalization tests were performed for different ground friction values and additional mass placed on the robot.

The final stage was the preparation of an adaptation module inspired by the RMA approach. This module learned to estimate hidden environment parameters based on the history of robot observations and actions, so that the policy does not need to receive privileged information directly during deployment.

### Teacher Mode, Friction 0.1

The `teacher` mode is a test variant in which the policy receives the true values of privileged observations, meaning the actual environment parameters such as ground friction and additional payload mass. This is a reference mode because the model does not need to estimate these values from the robot's movement history.

The tests were performed for a very low ground friction value of `0.1` with an additional payload of `2 kg`. Under these conditions, the robot maintained stable movement throughout the entire episode and no falls were recorded (`falls = 0`).

The plot shows that the velocity along the `x` axis quickly reaches the commanded value of `0.6 m/s` and then remains above the target. At the same time, a small lateral drift appears along the `y` axis, which is an expected effect at such low friction, but it does not lead to a loss of stability.

The friction and mass values in the privileged parameter plots remain consistent with the actual experiment configuration. This result shows that, with access to the true environment parameters, the policy can move correctly even on slippery ground.

![Teacher mode](images/teacher.png)

### Zero Mode, Friction 0.1

The `zero` mode is a test variant in which the policy receives neither the true privileged observations nor estimates from the adaptation module. Instead, the observation part corresponding to environment parameters is set to zero values. Therefore, this mode serves as a simple baseline showing how the policy behaves without correct information about friction and payload.

The tests were performed for a very low ground friction value of `0.1` with an additional payload of `2 kg`. Despite the lack of correct environment parameters, the robot maintained stable movement throughout the entire episode and no falls were recorded (`falls = 0`).

The plot shows that the velocity along the `x` axis quickly exceeds the commanded value of `0.6 m/s` and remains around `0.7 m/s` for most of the test. The lateral drift plot shows that the robot first deviates to one side, then corrects its motion and moves to the opposite side of the `y` axis. This differs from the `teacher` mode, where after the initial deviation the robot continued moving more steadily in one direction. This means that without the true privileged observation values, the policy still maintains balance, but direction correction is less smooth.

However, the parameter plots show a large mismatch between the values used by the model and the actual test conditions. The model behaves as if friction and payload mass had nominal values, while the actual friction is `0.1` and the additional mass is `2 kg`. This result shows that the base policy itself is quite robust, but the `zero` mode does not reflect the real environment parameters.

![Zero mode](images/zero.png)

### RMA Mode, Friction 0.1

The RMA mode is a test variant in which the policy does not receive the true privileged observation values directly. Instead, the adaptation module estimates hidden environment parameters, such as ground friction and additional payload mass, based on the history of robot observations and actions.

The tests were performed for a very low ground friction value of `0.1` with an additional payload of `2 kg`. Under these conditions, the robot maintained stable movement throughout the entire episode and no falls were recorded (`falls = 0`).

The plot shows that the velocity along the `x` axis quickly exceeds the commanded value of `0.6 m/s` and then remains above the target, similarly to the other modes. Lateral drift along the `y` axis is noticeable: the robot deviates to one side and only partially corrects its trajectory, but this does not lead to a loss of stability.

The most important difference compared to the `zero` mode is visible in the estimated parameter plots. The RMA module initially overestimates the friction value, but after a few seconds the estimate approaches the real friction value of `0.1`. The mass estimate also quickly reaches a value close to the real payload of `2 kg`. This means that the adaptation module correctly recognizes changed environment conditions and provides the policy with information close to the privileged observations used in the `teacher` mode.

![RMA mode](images/rma.png)

### Final Result

The most important result of the experiment is that the RMA module correctly brings the estimated friction close to `0.1` and the estimated mass close to `2 kg`. Thanks to this, the policy can act in a more informed way than in the `zero` mode, without requiring true privileged observations as in the `teacher` mode.

The RMA module was tested over a range of different environment parameters, including ground friction from `0.1` to `1.0` and additional robot payload from `0` to `2 kg`. In the conducted experiments, the estimates were not perfect from the first step, but they usually stabilized near the real values after a short time. This shows that the adaptation module can react to different environmental conditions.

## Bibliography

- RMA: Rapid Motor Adaptation for Legged Robots  
  https://arxiv.org/pdf/2107.04034

- Genesis: general-purpose robotics and embodied AI simulator  
  https://github.com/Genesis-Embodied-AI/Genesis

- Genesis locomotion example for Unitree Go2  
  https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_train.py

- RSL-RL: reinforcement learning library used for PPO training  
  https://github.com/leggedrobotics/rsl_rl

- Argo-Robot quadrupeds locomotion  
  https://github.com/Argo-Robot/quadrupeds_locomotion

---

## Team

- Artsiom Kruk
- Daniil Kavalevich
