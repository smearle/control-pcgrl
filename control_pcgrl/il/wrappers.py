import gym
import numpy as np
from control_pcgrl.configs.config import PoDConfig
from control_pcgrl.envs.reps.narrow_rep import NarrowRepresentation

from control_pcgrl.envs.reps.wrappers import RepresentationWrapper


class PoDWrapper(gym.Wrapper):
    """Wrapper for generating random paths of destruction."""
    def __init__(self, env, cfg: PoDConfig):
        self.env = env
        self.env.adjust_param(cfg=cfg)

        # Just for convenience. TODO: Port this logic to the environment itself if non-breaking.
        self.rep = self.unwrapped._rep
        self.prob = self.unwrapped._prob

        # Kind of a hack. Fine as long as we don't need to overwrite any of the init functionality in the representation
        # wrapper.
        # self.rep = PoDRepresentationWrapper(rep=self.rep, cfg=cfg)

        self.obfuscate_observation = cfg.obfuscate_observation

    # def queue_goal_map(self, goal_map: np.ndarray):
    #     self.unwrapped._rep.unwrapped._old_map = goal_map
    #     self.unwrapped._rep.unwrapped._random_start = False

    # def get_inverse_action_pair(self):
    #     destroy_action = self.action_space.sample()

    #     repair_action = self.get_repair_action()

    #     return destroy_action, repair_action 

    # def get_repair_action(self):

    #     # FIXME: Specific to narrow. Move this to representation classes.
    #     curr_tile_int = self.rep.unwrapped._map[tuple(self.rep.unwrapped._pos)]

    #     # Account no-op in narrow representation.
    #     # curr_tile_narrow_build_action = curr_tile_int + 1 

    #     repair_action = curr_tile_int

    #     return repair_action

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        return self._process_obs(obs), rew, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        return self._process_obs(obs), info

    def _process_obs(self, obs):
        if self.obfuscate_observation:
            obs = obfuscate_observation(obs)
        return obs


def obfuscate_observation(obs):
    """Remove everything but the padding. Fill the map with some tile."""
    obs_int = obs.argmax(-1)
    obs_int = np.where(obs_int != 0, 1, obs_int)
    obs = np.eye(obs.shape[-1])[obs_int]
    return obs

    
# class PoDRepresentationWrapper(RepresentationWrapper):
#     # FIXME: Move to (narrow) representation.
#     def __init__(self, rep, cfg):
#         super().__init__(rep, cfg)
#         self.rep.unwrapped.get_act_coords = self.get_act_coords

#     def get_act_coords(self):
#         act_coords = NarrowRepresentation.get_act_coords(self.rep)
#         act_coords = np.flip(act_coords, axis=0)
#         return act_coords