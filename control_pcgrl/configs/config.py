from hydra.core.config_store import ConfigStore
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict


@dataclass
class ModelConfig:
    name: Optional[str] = None
    conv_filters: Optional[int] = 64
    fc_size: Optional[int] = 64


@dataclass
class SeqNCAConfig(ModelConfig):
    name: str = "SeqNCA"
    conv_filters: int = 64    
    fc_size: int = 64
    patch_width: int = 3      # obs size of action branch of seqnca model, -1 for full obs

@dataclass
class ProblemConfig:
    name: str = MISSING
    weights: Dict[str, int] = MISSING
    controls: List[Any] = MISSING
    alp_gmm: bool = MISSING


@dataclass
class BinaryPathConfig(ProblemConfig):
    name: str = 'binary'
    # Regions weight will be 0 by default.
    weights: Dict[str, int] = field(default_factory = lambda: ({
    #    'player': 1,
    #    'create': 1,
    #    'target': 1,
    #    'regions': 1,
    #    'ratio': 1,
    #    'dist-win': 1,
    #    'sol-length': 2


        'path-length': 100,
    }))

@dataclass
class BinaryControlConfig(ProblemConfig):
    name: str = 'binary'
    weights: Dict[str, int] = field(default_factory = lambda: ({
        'path-length': 100,
        'regions': 100,
    }))
    controls: List[Any] = field(default_factory= lambda: [
        # 'regions',
        'regions', 'path-length',
    ])
    alp_gmm: bool = False


@dataclass
class MinecraftMazeConfig(ProblemConfig):
    name: str = "minecraft_3D_maze"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        'path-length': 100,
        'n_jump': 100,
        "regions": 0,
    }))


@dataclass
class MinecraftHoleyMazeConfig(ProblemConfig):
    name: str = "minecraft_3D_holey_maze"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "regions": 0,
        "path-length": 100,
        "connected-path-length": 120,
        "n_jump": 150,
    }))


@dataclass
class MinecraftHoleyDungeonConfig(ProblemConfig):
    name: str = "minecraft_3D_dungeon_holey"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "regions": 0, 
        "path-length": 100, 
        "chests": 300, 
        "n_jump": 100,
        "enemies": 100,
        "nearest-enemy": 200,
    }))


@dataclass
class MinecraftMazeControlConfig(ProblemConfig):
    weights: Dict[str, int] = field(default_factory = lambda: ({
        'path-length': 100,
        'n_jump': 100,
    }))
    controls: List[Any] = field(default_factory= lambda: [
        # 'regions',
        'n_jump', 'path-length',
    ])
    alp_gmm: bool = False


@dataclass
class MultiagentConfig:
    n_agents: int = MISSING
    # valid values: (shared, independent, JSON string)
    policies: str = "centralized"  # use shared weights by default


@dataclass
class SingleAgentConfig(MultiagentConfig):
    """Single agent environment etc."""
    n_agents: int = 0


@dataclass
class SingleAgentDummyMultiConfig(MultiagentConfig):
    """Multi-agent env and wrappers. Use this to validate our multiagent implementation."""
    n_agents: int = 1


@dataclass
class SharedPolicyConfig(MultiagentConfig):
    n_agents: int = 2


@dataclass
class HardwareConfig:
    n_cpu: int = MISSING
    n_gpu: int = MISSING
    num_envs_per_worker: int = 60

@dataclass
class LocalHardwareConfig(HardwareConfig):
    n_cpu: int = 3
    n_gpu: int = 1

@dataclass
class RemoteHardwareConfig(HardwareConfig):
    n_cpu: int = 12
    n_gpu: int = 1



@dataclass
class ControlPCGRLConfig:
    # Specify defaults for sub-configs so that we can override them on the command line. (Whereas we can cl-override 
    # other settings as-is.)
    defaults: List[Any] = field(default_factory=lambda: [
        {'problem': 'minecraft_3D_maze'},
        {'hardware': 'remote'},
        # TODO: Picking the default should happen here, in the configs, instead of in the main code, perhaps.
        {'model': 'default_model'},
        {'multiagent': 'single_agent'},
        '_self_',
    ])

    # If you specify defaults here there will be problems when you try to overwrite on CL.
    hardware: HardwareConfig = MISSING
    model: ModelConfig = MISSING
    multiagent: MultiagentConfig = MISSING
    problem: ProblemConfig = MISSING

    algorithm: str = 'PPO'
    debug: bool = False
    render: bool = False
    infer: bool = False
    evaluate: bool = False
    load: bool = True
    overwrite: bool = False
    wandb: bool = False

    exp_id: str = '0'
    representation: str = 'narrow'
    show_agents: bool = False
    learning_rate: float = 5e-6
    gamma: float = 0.99
    map_shape: List[Any] = field(default_factory=lambda: # this is not right to hard code it here my friend? why do we need this?
        [16, 16]
    )
    # map_shape: Optional[List[Any]] = None
    # crop_shape: Optional[List[Any]] = None 
    crop_shape: List[Any] = field(default_factory=lambda: 
        [32, 32]
    )
    max_board_scans: int = 3
    n_aux_tiles: int = 0
    observation_size: Optional[int] = None
    controls: Optional[ProblemConfig] = None
    change_percentage: Optional[float] = None
    static_prob: Optional[float] = None
    action_size: Optional[List[Any]] = None
    # action_size: List[Any] = field(default_factory=lambda: 
    #     [3, 3]
    # )

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

cs.store(name="single_agent", group="multiagent", node=SingleAgentConfig)
cs.store(name="single_agent_dummy_multi", group="multiagent", node=SingleAgentDummyMultiConfig)
cs.store(name="shared_policy", group="multiagent", node=SharedPolicyConfig)

cs.store(name="binary_path", group="problem", node=BinaryPathConfig)
cs.store(name="binary_path_control", group="problem", node=BinaryControlConfig)

cs.store(name="minecraft_3D_maze", group="problem", node=MinecraftMazeConfig)
cs.store(name="minecraft_3D_holey_maze", group="problem", node=MinecraftHoleyMazeConfig)
cs.store(name="minecraft_3D_dungeon_holey", group="problem", node=MinecraftHoleyDungeonConfig)

cs.store(name="default_model", group="model", node=ModelConfig)
cs.store(name="seqnca", group="model", node=SeqNCAConfig)
