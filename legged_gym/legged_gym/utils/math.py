# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, quat_mul, quat_conjugate
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    """Rotate a vector only around the yaw-direction.

    Args:
        quat (torch.Tensor): Input orientation to extract yaw from.
        vec (torch.Tensor): Input vector.

    Returns:
        torch.Tensor: Rotated vector.
    """
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Implements box-minur operator (quaternion difference)
    https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf

    Args:
        q1 (torch.Tensor): quaternion
        q2 (torch.Tensor): quaternion

    Returns:
        torch.Tensor: q1 box-minus q2
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    re = quat_diff[:, -1]  # real part, q = [x, y, z, w] = [re, im]
    im = quat_diff[:, 0:3]  # imaginary part
    norm_im = torch.norm(im, dim=1)
    scale = 2.0 * torch.where(norm_im > 1.0e-7, torch.atan(norm_im / re) / norm_im, torch.sign(re))
    return scale.unsqueeze(-1) * im

# @ torch.jit.script
def wrap_to_pi(angles):
    """Wraps input angles (in radians) to the range [-pi, pi].

    Args:
        angles (torch.Tensor): Input angles.

    Returns:
        torch.Tensor: Angles in the range [-pi, pi].
    """
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    """Randomly samples tensor from a triangular distribution.

    Args:
        lower (float): The lower range of the sampled tensor.
        upper (float): The upper range of the sampled tensor.
        size (Tuple[int, int]): The shape of the tensor.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled tensor of shape :obj:`size`.
    """
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)