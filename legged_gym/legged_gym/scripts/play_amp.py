# !/usr/bin/env python3
# Code from 
#   Adversarial Motion Priors Make Good Substitutes for  Complex Reward Functions 
#       2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
# Paper: https://arxiv.org/abs/2203.15103
# Project: -
# Author: Alejandro Escontrela, Xue Bin Peng, Wenhao Yu, Tingnan Zhang  Atil Iscen, Ken Goldberg, Pieter Abbeel
# Affiliation: UC Berkeley, Google Brain

# import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    img_idx = 0

    video_duration = 10
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    for i in range(num_frames):
        actions = policy(obs.detach())
        obs, _, _, _, infos, _, _ = env.step(actions.detach())

        # Reset camera position.
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
        camera_relative_position = 1.2 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
        env.set_camera(look_at + camera_relative_position, look_at)

        # if RECORD_FRAMES:
        #     frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
        #     if not os.path.isdir(frames_path):
        #         os.mkdir(frames_path)
        #     filename = os.path.join('logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
        #     env.gym.write_viewer_image_to_file(env.viewer, filename)
        #     img = cv2.imread(filename)
        #     if video is None:
        #         video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
        #     video.write(img)
        #     img_idx += 1 

    video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    # RECORD_FRAMES = False
    args = get_args()
    play(args)