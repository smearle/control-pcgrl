from typing import Any, List, Optional
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    pass


@dataclass
class ControlConfig:
    controls: List[Any] = field(default_factory=[
        'regions', 'path-length',
    ])
    alp_gmm: bool = False


@dataclass
class MutiagentConfig:
    n_agents: int = 2


@dataclass
class PCGRLConfig:
    exp_id: str = '0'
    problem: str = 'binary'
    retpresentation: str = 'turtle'
    model: Optional[ModelConfig] = None
    # controls: List[Any] = field(default_factory=lambda: [
    # ])
    learning_rate: float = 5e-6
    max_board_scans: int = 3
    n_aux_tiles: int = 0
    observation_size: Optional[int] = None
    controls: Optional[ControlConfig] = None
    change_percentage: Optional[float] = None
    crop_shape: Optional[List[Any]] = field(default_factory=lambda:
        []
    )
    static_prob: Optional[float] = None
    action_size: Optional[List[Any]] = None
    multiagent: Optional[MutiagentConfig] = None


cs = ConfigStore.instance()
# Registering the Config class with the name `postgresql` with the config group `db`
cs.store(name="pcgrl", node=PCGRLConfig)
