# Isaac Gym 具身智能与仿生机器人环境 (Embodied AI & Bio-Robotics)

这个仓库是基于 **NVIDIA Isaac Gym** 的强化学习（RL）环境，专门用于训练​**仿生足式机器人**​（如四足、双足）在**复杂崎岖地形**上行走。

它在原版 `legged_gym` 和 `rsl_rl` 代码的基础上，​**整合了具身智能 (Embodied AI) 领域的先进算法**​，如 **WASABI** 等，并提供了即插即用的​**模块化强化学习框架**​，旨在促进 Sim-to-Real 迁移的研究和应用。

该仓库的核心价值在于提供了一个强大的平台，让研究人员能够：

* 🚀 **快速验证和部署**最前沿的具身智能算法。
* 🤖 **训练**包括 Unitree G1/H1 在内的​**多种足式机器人**​。
* 🌐 ​**实现高保真度的模拟到现实 (Sim-to-Real) 迁移**​，集成了执行器网络、摩擦/质量随机化、噪声观测和训练期间的随机扰动等关键技术。

仓库目录结构和功能介绍如下：

```
Embodied_related   # 具身智能相关库
│
├── rsl_rl/rsl_rl        # ​模块化强化学习框架​，负责具身智能算法的具体实现、数据处理和训练逻辑
│    ├─ algorithms               # 具体算法实现
│    │     └─                           # 目前支持 PPO 和 WASABI 算法
│    ├─ datasets                 # 离线算法的数据集（一般是动作模型的数据）
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
├── legged_gym           # 负责 Isaac Gym 环境和智能体（机器人模型）的实现
│    ├─ resources
│    │     ├─ dataset_loader  # 机器人加载动作模型逻辑，如 WASABI 算法
│    │     └─ robots               # 机器人仿真所需文件，如 dae 和 urdf 或 meshes
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

支持机器人的类型如下：

|机器人类型|模型|制造商/来源|特点|
| --- | --- | --- | --- |
|**四足** (Quadruped)|ANYmal B/C|ANYbotics|高性能四足机器人，适用于极限环境|
|**四足** (Quadruped)|Solo8, A1|Unitree|紧凑型/通用型四足机器人|
|**四足** (Quadruped)|G1, Go2|Unitree|常用四足机器人系列|
|**双足** (Humanoid)|H1, H1\_2|Unitree|Unitree 仿人机器人系列|
|**双足** (Humanoid)|Cassie|Agility Robotics|动态双足行走平台|

## 快速开始

### 🚀 训练一个任务

​**最新进展 (2025/10/09)**​：已实现 WASABI 算法的集成，并注册为 `solo8` 任务。

您可以使用以下命令开始训练：

```bash
python scripts/train.py --task solo8 --headless
```

**最新进展 (2025/10/10)**​：已实现 腾讯举办的2025年开悟比赛具身智能赛道复赛环境 的集成，并注册为 `unitree_go2_comp_s2` 任务。

您可以使用以下命令开始训练：

```bash
python scripts/train.py --task unitree_go2_comp_s2 --headless
```

### 📈 Tensorboard 监控面板（远程访问）

如果您在远程服务器上进行训练，可以通过 SSH 端口转发在本地浏览器中查看训练曲线：

1. ​**在服务器 shell 中运行 Tensorboard**​：

```bash
tensorboard --logdir=logs/flat_solo8/Oct09_20-54-04_wasabi
```

2. ​**在本地 Git Bash/终端中设置 SSH 端口转发**​（假设服务器 IP 为 `192.168.2.101`，端口为 `30441`）：

```bash
ssh -N -f -L 6006:localhost:6006 root@192.168.2.101 -p 30441
```

输入密码后，接口监听成功。

3. ​**在本地浏览器打开**​：

```bash
<http://localhost:6006/>
```

### 集成case1：offline算法，AC+GAN结构

在集成过程中，根据 WASABI 的经验，需要修改以下地方：

```
Embodied_related   # 具身智能相关库
│
├── rsl_rl/rsl_rl        # 具身智能算法的实现
│    ├─ algorithms               # 具体算法实现
│    │     └─                           # 加入 WASABI 算法
│    ├─ datasets                 # 离线算法的数据集（一般是动作模型的数据）
│    │     └─                           # 目前对 WASABI 算法做适配
│    ├─ modules                  # 强化学习的一些模块
│    │     └─                           # 比如 Actor-Critic结构、递归Actor-Critic结构、判别器等模块
│    └─ runners                  # 强化学习训练过程的辅助函数
│          └─                           # 比如 learn、log、save、load等函数
├── legged_gym           # 具身智能智能体和环境的实现
│    ├─ resources
│    │     ├─ dataset_loader  # 机器人加载动作模型逻辑，如 WASABI 算法
│    │     └─ robots               # 机器人仿真所需文件，如 dae 和 urdf 或 meshes
│    ├─ legged_gym
│    │     ├─ envs              # 机器人与环境交互逻辑、该机器人算法参数、环境参数以及任务注册的逻辑
│    │     │    ├─ __init__.py          # 该目录下所有任务注册的逻辑
│    │     │    └─                      # 具体机器人与环境交互逻辑、机器人算法参数、环境参数
│    │     ├─ scripts           # 训练和演示目录
│    │     │    └─ play.py              # 演示
│    │     └─ utils             # 辅助函数，包括数学、日志、注册任务实现逻辑、地形生成逻辑
```

