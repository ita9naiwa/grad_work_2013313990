import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)

register(
    id='cluster-scheduling',
    entry_point='clutser-scheduling.envs:Mao',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=False
)
 
