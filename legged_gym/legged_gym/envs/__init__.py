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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .base.legged_robot_parkour import LeggedRobot as LeggedRobotParkour
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .solo8_wasabi.solo8 import Solo8
from .solo8_wasabi.solo8_config import Solo8FlatCfg, Solo8FlatCfgPPO
from .unitree_a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .unitree_a1_amp.legged_robot_amp import A1AMPRobot
from .unitree_a1_amp.a1_config_amp import A1AMPCfg, A1AMPCfgPPO
from .unitree_a1_parkour.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from .unitree_a1_parkour.a1_field_distill_config import A1FieldDistillCfg, A1FieldDistillCfgPPO
from .unitree_g1.g1_env import G1Robot
from .unitree_g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from .unitree_g1_parkour.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from .unitree_g1_parkour.go1_field_distill_config import Go1FieldDistillCfg, Go1FieldDistillCfgPPO
from .unitree_go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .unitree_go2_comp_s2.go2_config_comp_KaiWu_s2 import GO2KaiWuS2RoughCfg, GO2KaiWuS2RoughCfgPPO
from .unitree_go2_parkour.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from .unitree_go2_parkour.go2_distill_config import Go2DistillCfg, Go2DistillCfgPPO
from .unitree_go2_comp_s2.go2_env_comp_KaiWu_s2 import Go2KaiWuS2
from .unitree_h1.h1_env import H1Robot
from .unitree_h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from .unitree_h1_2.h1_2_env import H1_2Robot
from .unitree_h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "unitree_a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "unitree_g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO() )
task_registry.register( "unitree_go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO() )
task_registry.register( "unitree_h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO() )
task_registry.register( "unitree_h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO() )


task_registry.register( "solo8_wasabi", Solo8, Solo8FlatCfg(), Solo8FlatCfgPPO() )
task_registry.register( "unitree_a1_amp", A1AMPRobot, A1AMPCfg(), A1AMPCfgPPO() )
task_registry.register( "unitree_a1_parkour_field", LeggedRobotParkour, A1FieldCfg(),A1FieldCfgPPO())
task_registry.register( "unitree_a1_parkour_distill", LeggedRobotParkour, A1FieldDistillCfg(), A1FieldDistillCfgPPO())
task_registry.register( "unitree_g1_parkour_field", LeggedRobotParkour, Go1FieldCfg(), Go1FieldCfgPPO())
task_registry.register( "unitree_g1_parkour_distill", LeggedRobotParkour, Go1FieldDistillCfg(), Go1FieldDistillCfgPPO())
task_registry.register( "unitree_go2_comp_s2", Go2KaiWuS2, GO2KaiWuS2RoughCfg(), GO2KaiWuS2RoughCfgPPO() )
task_registry.register( "unitree_go2_parkour_field", LeggedRobotParkour, Go2FieldCfg(), Go2FieldCfgPPO())
task_registry.register( "unitree_go2_parkour_distill", LeggedRobotParkour, Go2DistillCfg(), Go2DistillCfgPPO())
