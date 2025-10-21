""" A file put all mixin class combinations """
from .actor_critic_parkour import ActorCritic
from .actor_critic_recurrent_parkour import ActorCriticRecurrent
from .encoder_actor_critic_parkour import EncoderActorCriticMixin
from .state_estimator_parkour import EstimatorMixin

class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass

class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    
    def load_misaligned_state_dict(self, module, obs_segments, privileged_obs_segments=None):
        pass