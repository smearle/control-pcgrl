import hydra

from control_pcgrl.configs.config import Config
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config


@hydra.main(config_path="control_pcgrl/configs", config_name="config")
def test_render(cfg: Config):
    validate_config(cfg)
    env = make_env(cfg)

    while True:
        done = False
        obs = env.reset()
        # Randomly sample actions
        for i in range(100):
            obs, rew, done, truncated, info = env.step(env.action_space.sample())
            env.render()


if __name__ == '__main__':
    test_render()