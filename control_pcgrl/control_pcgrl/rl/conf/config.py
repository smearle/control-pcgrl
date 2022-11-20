from hydra.core.config_store import ConfigStore
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import MISSING, dataclass, field


@dataclass
class ModelConfig:
    pass


@dataclass
class ControlConfig:
    contrls: List[Any] = MISSING
    alp_gmm: bool = MISSING

@dataclass
class BinaryControlConfig(ControlConfig):
    controls: List[Any] = field(default_factory= lambda: [
        # 'regions',
        'regions', 'path-length',
    ])
    alp_gmm: bool = False


@dataclass
class MultiagentConfig:
    n_agents: int = MISSING

@dataclass
class TwoAgentConfig(MultiagentConfig):
    n_agents: int = 2


@dataclass
class HardwareConfig:
    n_cpu: int = MISSING
    n_gpu: int = MISSING
    num_envs_per_worker: int = 10

@dataclass
class LocalHardwareConfig(HardwareConfig):
    n_cpu: int = 1
    n_gpu: int = 0

@dataclass
class RemoteHardwareConfig(HardwareConfig):
    n_cpu: int = 8
    n_gpu: int = 1


# Register hardware configs as group


@dataclass
class ControlPCGRLConfig:
    debug: bool = False
    render: bool = False
    infer: bool = False
    evaluate: bool = False
    load: bool = True
    overwrite: bool = False
    hardware: HardwareConfig = LocalHardwareConfig()
    wandb: bool = False

    exp_id: str = '0'
    problem: str = 'binary'
    representation: str = 'turtle'
    model: Optional[ModelConfig] = None
    learning_rate: float = 5e-6
    gamma: float = 0.99
    map_shape: List[Any] = field(default_factory=lambda: 
        [16, 16]
    )
    crop_shape: List[Any] = field(default_factory=lambda: 
        [32, 32]
    )
    max_board_scans: int = 3
    n_aux_tiles: int = 0
    observation_size: Optional[int] = None
    controls: Optional[ControlConfig] = None
    change_percentage: Optional[float] = None
    static_prob: Optional[float] = None
    action_size: Optional[List[Any]] = None
    # action_size: List[Any] = field(default_factory=lambda: 
    #     [3, 3]
    # )
    multiagent: Optional[MultiagentConfig] = None
    # multiagent: Optional[MultiagentConfig] = TwoAgentConfig()

    # Gets set later :)
    log_dir: Optional[Path] = None
    env_name: Optional[str] = None

    # This one (and other stuff) could be in a separate PCGRLEnvConfig
    evaluation_env: Optional[bool] = None


cs = ConfigStore.instance()
# Registering the Config class with the name `postgresql` with the config group `db`
cs.store(name="pcgrl", node=ControlPCGRLConfig)

cs.store(name="local", group="hardware", node=LocalHardwareConfig)
cs.store(name="remote", group="hardware", node=RemoteHardwareConfig)

cs.store(name="two_agent", group="multiagent", node=TwoAgentConfig)

cs.store(name="binary_control", group="controls", node=BinaryControlConfig)