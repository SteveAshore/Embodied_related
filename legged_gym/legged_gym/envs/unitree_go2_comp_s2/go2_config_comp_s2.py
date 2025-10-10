from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2S2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 48  # robot state (48)
        num_privileged_obs = 259  # robot state (48) + height scans (17*11=187) 
        height_dim = 187
        privileged_dim = 27
        history_length = 1
        num_actions = 12  # joint positions, velocities or torques
        episode_length_s = 20 # episode length in seconds
        privileged_obs = True
        test = False
        algo = "ppo"
        
    class terrain( LeggedRobotCfg.terrain ):
        border_size = 50 # [m]
        curriculum = True
        max_init_terrain_level = 9 # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [ripples, slope, stairs_up, stairs_down, obstacles, flat]
        terrain_proportions = [0.1, 0.05, 0.4, 0.2, 0.2, 0.0, 0.0, 0.05]
        is_eval = False
        num_sub_terrains = 200

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_lin_vel_x_curriculum = 1
        max_lin_vel_y_curriculum = 0.2
        max_ang_vel_yaw_curriculum = 0.5
        max_flat_lin_vel_x_curriculum = 1.0
        max_flat_lin_vel_y_curriculum = 1.0
        max_flat_ang_vel_yaw_curriculum = 3.0
        class ranges:
            lin_vel_x = [-0.4, 0.4] # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            heading = [-0.785, 0.785]
            flat_lin_vel_x = [-0.4, 0.4] # min max [m/s]
            flat_lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            flat_ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            flat_heading = [-0.785, 0.785]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_restitution = True
        randomize_com_pos = True
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [0.05, 2.75]
        restitution_range = [0.0, 1.0]
        added_mass_range = [0.0, 3.0]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        com_pos_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        randomize_obs_latency = False
        latency_range = [0.0, 0.02]
        push_interval = 751.0

    class rewards( LeggedRobotCfg.rewards ):
        tracking_sigma = 0.1
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1
        soft_torque_limit = 1
        base_height_target = 0.25
        foot_height_target = 0.0
        max_contact_force = 100
        only_positive_rewards = False
        reward_curriculum = True
        reward_curriculum_term = []
        reward_curriculum_schedule = [0, 1000, 1, 0]
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.2
            lin_vel_z = -0.0
            ang_vel_xy = -0.05
            orientation = -0.9
            torques = -0.0005
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time =  0.7
            collision = -0.1
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            terrain_adaptation = 0.3
            energy_efficiency = 0.05
            obstacle_clearance = 0.6
            forward_progress = 1.6
            
    class normalization( LeggedRobotCfg.normalization ):
        clip_actions = 6

    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0

class GO2S2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 3e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'roughGo2'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

  
