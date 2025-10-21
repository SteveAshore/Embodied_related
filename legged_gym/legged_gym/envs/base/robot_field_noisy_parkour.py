
from legged_gym.legged_gym.envs.base.legged_robot_field_parkour import LeggedRobotField
from legged_gym.legged_gym.envs.base.legged_robot_noisy_parkour import LeggedRobotNoisyMixin

class RobotFieldNoisy(LeggedRobotNoisyMixin, LeggedRobotField):
    """ Using inheritance to combine the two classes.
    Then, LeggedRobotNoisyMixin and LeggedRobot can also be used elsewhere.
    """
    pass
