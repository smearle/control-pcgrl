import hydra
from control_pcgrl.configs.config import EvalConfig
from control_pcgrl.rl.train_ctrl import main as train_main

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: EvalConfig):
    assert cfg.evaluate is True
    # For now just use the eval block in our main training function
    train_main(cfg)
