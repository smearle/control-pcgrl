import hydra
from control_pcgrl.configs.config import EnjoyConfig
from control_pcgrl.rl.train import main as train_main

@hydra.main(version_base="1.3", config_path="../configs", config_name="enjoy")
def main(cfg: EnjoyConfig):
    assert cfg.infer == True
    # For now just use the eval block in our main training function
    train_main(cfg)
