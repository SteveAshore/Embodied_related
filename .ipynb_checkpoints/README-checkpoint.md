# Isaac Gym Environments for Legged Robots

该repo尝试集成目前具身智能的一些方法，包含`legged_gym`和`rsl_rl`两个核心库。目前项目的结构如下：

```
Embodied_related   # 具身智能相关库
│
├── rsl_rl/rsl_rl        # 具身智能算法的实现
│    ├─ algorithms               # 具体算法实现
│    │     └─                           # 目前支持 PPO 和 WASABI 算法
│    ├─ dataset_loader           # 加载motion.pt
│    │     └─                           # 目前对 WASABI 算法做适配
│    ├─ datasets                 # 离线算法的数据集
│    │     └─                           # 目前对 WASABI 算法做适配
│    ├─ env                      # 强化学习环境
│    │     └─                           # 环境基类
│    ├─ modules                  # 强化学习的一些模块
│    │     └─                           # 比如 Actor-Critic结构、递归Actor-Critic结构、判别器等模块
│    ├─ runners                  # 强化学习训练过程的辅助函数
│    │     └─                           # 比如 learn、log、save、load等函数
│    ├─ storage                  # 强化学习的缓冲池
│    │     └─                           # 比如 buffer 和 小型的 RolloutStorage
│    └─ utils                    # 强化学习操作trajectory
│          └─                           # 比如 split、pad 和 unpad 操作
├── legged_gym           # 具身智能智能体和环境的实现
│    ├─ resources
│    │     ├─ datasets          # 机器人预训练好的模型，如 WASABI 算法
│    │     └─ robots            # 机器人仿真所需文件，如 dae 和 urdf 或 meshes
│    ├─ legged_gym
│    │     ├─ __init__.py       # 初始化根目录
│    │     ├─ envs              # 机器人与环境交互逻辑、该机器人算法参数、环境参数以及任务注册的逻辑
│    │     │    ├─ __init__.py          # 该目录下所有任务注册的逻辑
│    │     │    ├─                      # 具体机器人与环境交互逻辑、机器人算法参数、环境参数
│    │     │    └─ base                 # 基类机器人与环境交互逻辑、机器人算法参数、环境参数
│    │     ├─ scripts           # 训练和演示目录
│    │     │    ├─ train.py             # 训练
│    │     │    └─ play.py              # 演示
│    │     ├─ tests             # 测试脚本
│    │     │    └─ test_env.py          # 测试10轮运行是否正常
│    │     └─ utils             # 辅助函数，包括数学、日志、注册任务实现逻辑、地形生成逻辑
```

目前进展如下：

### 20251009

实现`wasabi`算法的集成，注册为`solo8`任务，可通过下面的命令训练：

```bash
python scripts/train.py --task solo8 --headless
```

### tensorboard面板本地浏览器查看方法：

1. 服务器开发环境下在shell运行命令：

```bash
tensorboard --logdir=logs/flat_solo8/Oct09_20-54-04_wasabi
```

1. 本地打开 git bash，输入命令：

```bash
ssh -N -f -L 6006:localhost:6006 root@192.168.2.101 -p 30441
```

然后随后输入密码，如果弹出默认消息，则已实现接口监听。

1. 本地打开网址：

```bash
http://localhost:6006/
```

即可显示监控面板

This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.

**Maintainer**: Nikita Rudin

**Affiliation**: Robotic Systems Lab, ETH Zurich

**Contact**: [rudinn@ethz.ch](mailto:rudinn@ethz.ch)

---

### :bell: Announcement (09.01.2024)

With the shift from Isaac Gym to Isaac Sim at NVIDIA, we have migrated all the environments from this work to [Isaac Lab](https://github.com/isaac-sim/IsaacLab). Following this migration, this repository will receive limited updates and support. We encourage all users to migrate to the new framework for their applications.

Information about this work's locomotion-related tasks in Isaac Lab is available [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#locomotion).

---

### Useful Links

Project website: https://leggedrobotics.github.io/legged_gym/

Paper: https://arxiv.org/abs/2109.11978

### Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f <https://download.pytorch.org/whl/cu113/torch_stable.html`>
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
    - Try running an example `cd examples && python 1080_balls_of_solitude.py`
    - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
    - Clone https://github.com/leggedrobotics/rsl_rl
    - `cd rsl_rl && git checkout v1.0.2 && pip install -e .`
5. Install legged_gym
    - Clone this repository
    - `cd legged_gym && pip install -e .`

### CODE STRUCTURE

1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).
2. Both env and config classes use inheritance.
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.

### Usage

1. Train:`python legged_gym/scripts/train.py --task=anymal_c_flat`
    - To run on CPU add following arguments: `-sim_device=cpu`, `-rl_device=cpu` (sim on CPU and rl on GPU is possible).
    - To run headless (no rendering) add `-headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    - The following command line arguments override the values set in the config files:
    - -task TASK: Task name.
    - -resume: Resume training from a checkpoint
    - -experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
    - -run_name RUN_NAME: Name of the run.
    - -load_run LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
    - -checkpoint CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
    - -num_envs NUM_ENVS: Number of environments to create.
    - -seed SEED: Random seed.
    - -max_iterations MAX_ITERATIONS: Maximum number of training iterations.
2. Play a trained policy:`python legged_gym/scripts/play.py --task=anymal_c_flat`
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment

The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales.

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!

### Troubleshooting

1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`. It is also possible that you need to do `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(for conda user. Replace /path/to/ to the corresponding path.).

### Known Issues

1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesirable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from the reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:

```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.

```