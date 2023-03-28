from control_pcgrl.configs.config import PoDConfig
from control_pcgrl.il.wrappers import PoDWrapper
from control_pcgrl.rl.envs import make_env


def make_pod_env(cfg: PoDConfig):
    env = make_env(cfg)
    env = PoDWrapper(env, cfg)
    return env