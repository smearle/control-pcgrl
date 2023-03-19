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
class TaskConfig:
    name: str = MISSING
    problem: str = MISSING
    weights: Dict[str, int] = MISSING
    controls: List[Any] = MISSING
    alp_gmm: bool = False
    map_shape: List[Any] = MISSING
    obs_window: List[Any] = MISSING


# @dataclass
# class BinaryConfig(ProblemConfig):
#     problem: str = 'binary'
#     # Regions weight will be 0 by default.
#     weights: Dict[str, int] = field(default_factory = lambda: ({
#         'path-length': 100,
#         'regions': 100,
#     }))
#     map_shape: List[Any] = field(default_factory= lambda: [16, 16])
#     crop_shape: List[Any] = field(default_factory= lambda: [32, 32])


# @dataclass
# class BinaryPathConfig(BinaryConfig):
#     weights: Dict[str, int] = field(default_factory = lambda: ({
#         'path-length': 100,
#     }))


# @dataclass
# class BinaryControlConfig(BinaryConfig):
#     controls: List[Any] = field(default_factory= lambda: [
#         # 'regions',
#         'regions', 'path-length',
#     ])
#     alp_gmm: bool = False

# @dataclass
# class ZeldaConfig(TaskConfig):
#     problem: str = 'zelda'
#     map_shape: List[Any] = field(default_factory= lambda: [16, 16])
#     crop_shape: List[Any] = field(default_factory= lambda: [32, 32])
#     weights: Dict[str, int] = field(default_factory = lambda: ({
#         "player": 3,
#         "key": 3,
#         "door": 3,
#         "regions": 5,
#         "enemies": 1,
#         "nearest-enemy": 2,
#         "path-length": 1
#     }))


# @dataclass
# class ZeldaControlConfig(ZeldaConfig):
#     controls: List[Any] = field(default_factory= lambda: [
#         # 'path-length',
#         'nearest-enemy', 'path-length',
#     ])
#     alp_gmm: bool = False

@dataclass
class SokobanConfig(TaskConfig):
    problem: str = 'sokoban'
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "player": 3,
        "crate": 2,
        "target": 2,
        "regions": 5,
        "ratio": 2,
        "dist-win": 0,
        "sol-length": 1,
    }))


@dataclass
class SokobanControlConfig(SokobanConfig):
    controls: List[Any] = field(default_factory= lambda: [
        # 'crate',
        'crate', 'sol-length',
    ])
    alp_gmm: bool = False


@dataclass
class SMBConfig(TaskConfig):
    problem: str = 'smb'
    # NOTE that the map_shape and crop_shape are y, x here
    map_shape: List[Any] = field(default_factory= lambda: [116, 16])
    crop_shape: List[Any] = field(default_factory= lambda: [232, 32])
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "dist-floor": 2,
        "disjoint-tubes": 1,
        "enemies": 1,
        "empty": 1,
        "noise": 4,
        "jumps": 2,
        "jumps-dist": 2,
        "dist-win": 5,
        "sol-length": 1,
    }))


@dataclass
class SMBControlConfig(SMBConfig):
    controls: List[Any] = field(default_factory= lambda: [
        # 'dist-floor',
        'dist-floor', 'sol-length',
    ])
    alp_gmm: bool = False


@dataclass
class LegoConfig(TaskConfig):
    problem: str = 'lego'
    # NOTE that the map_shape and crop_shape are z, y, x here
    map_shape: List[Any] = field(default_factory= lambda: [10, 10, 10])
    crop_shape: List[Any] = field(default_factory= lambda: [20, 20, 20])
    weights: Dict[str, int] = field(default_factory = lambda: ({
        'n_bricks': 1,
    }))


@dataclass
class MinecraftConfig(TaskConfig):
    problem: str = MISSING
    map_shape: List[Any] = field(default_factory= lambda: [15, 15, 15])
    crop_shape: List[Any] = field(default_factory= lambda: [30, 30, 30])


@dataclass
class MinecraftMazeConfig(MinecraftConfig):
    problem: str = "minecraft_3D_maze"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        'path-length': 100,
        'n_jump': 100,
        "regions": 0,
    }))


@dataclass
class MinecraftHoleyMazeConfig(MinecraftConfig):
    problem: str = "minecraft_3D_holey_maze"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "regions": 0,
        "path-length": 100,
        "connected-path-length": 120,
        "n_jump": 150,
    }))


@dataclass
class MinecraftHoleyDungeonConfig(MinecraftConfig):
    problem: str = "minecraft_3D_dungeon_holey"
    weights: Dict[str, int] = field(default_factory = lambda: ({
        "regions": 0, 
        "path-length": 100, 
        "chests": 300, 
        "n_jump": 100,
        "enemies": 100,
        "nearest-enemy": 200,
    }))


@dataclass
class MinecraftMazeControlConfig(MinecraftConfig):
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
    n_envs_per_worker: int = 2

@dataclass
class LocalHardwareConfig(HardwareConfig):
    n_cpu: int = 2
    n_gpu: int = 1

@dataclass
class RemoteHardwareConfig(HardwareConfig):
    n_cpu: int = 12
    n_gpu: int = 1
    n_envs_per_worker: int = 20


@dataclass
class Config:
    # Specify defaults for sub-configs so that we can override them on the command line. (Whereas we can cl-override 
    # other settings as-is.)
    defaults: List[Any] = field(default_factory=lambda: [
        {'task': 'binary'},
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
    task: TaskConfig = MISSING
    train_reward_model: bool = False

    algorithm: str = 'PPO'
    debug: bool = False
    render: bool = False
    render_mode: str = 'human'
    infer: bool = False
    evaluate: bool = False
    load: bool = True
    overwrite: bool = False
    wandb: bool = False
    timesteps_total: int = int(1e10)

    exp_id: str = '0'
    representation: str = 'narrow'
    show_agents: bool = False  # Represent agent(s) on the map using an additional observation channel
    learning_rate: float = 5e-6
    gamma: float = 0.99
    max_board_scans: int = 3
    n_aux_tiles: int = 0
    controls: Optional[TaskConfig] = None
    change_percentage: Optional[float] = None
    static_prob: Optional[float] = None
    act_window: Optional[List[Any]] = None
    # action_size: List[Any] = field(default_factory=lambda: 
    #     [3, 3]
    # )

    # Will default to `control-pcgrl/rl_runs` if not specified.
    runs_dir: Optional[Path] = None

    # Gets set later :)
    log_dir: Optional[Path] = None
    env_name: Optional[str] = None

    # This one (and other stuff) could be in a separate PCGRLEnvConfig
    evaluation_env: Optional[bool] = None

    n_eval_episodes: int = 1


@dataclass
class EnjoyConfig(Config):
    """Config for enjoying."""
    # Indicate that we cannot overwrite this
    render: bool = True
    infer: bool = True

    render_mode: str = 'human'
    # render_mode: str = 'gtk'


@dataclass
class EvalConfig(Config):
    """Config for evaluation."""
    # Indicate that we cannot overwrite this
    evaluate: bool = True

    n_eval_episodes: int = 1

@dataclass
class CrossEvalConfig(EvalConfig):
    """Config for cross-evaluation."""
    name: str = "lr_sweep"


cs = ConfigStore.instance()
# Registering the Config class with the name `postgresql` with the config group `db`
cs.store(name="train", node=Config)
cs.store(name="enjoy", node=EnjoyConfig)
cs.store(name="eval", node=EvalConfig)
cs.store(name="cross_eval", node=CrossEvalConfig)

cs.store(name="local", group="hardware", node=LocalHardwareConfig)
cs.store(name="remote", group="hardware", node=RemoteHardwareConfig)

cs.store(name="single_agent", group="multiagent", node=SingleAgentConfig)
cs.store(name="single_agent_dummy_multi", group="multiagent", node=SingleAgentDummyMultiConfig)
cs.store(name="shared_policy", group="multiagent", node=SharedPolicyConfig)

cs.store(name="base_task", group="task", node=TaskConfig)
# cs.store(name="binary", group="problem", node=BinaryConfig)
# cs.store(name="binary_path", group="problem", node=BinaryPathConfig)
# cs.store(name="binary_control", group="problem", node=BinaryControlConfig)

# cs.store(name="zelda", group="task", node=ZeldaConfig)
# cs.store(name="zelda_control", group="task", node=ZeldaControlConfig)

cs.store(name="sokoban", group="problem", node=SokobanConfig)
cs.store(name="sokoban_control", group="task", node=SokobanControlConfig)

cs.store(name="smb", group="task", node=SMBConfig)
cs.store(name="smb_control", group="task", node=SMBControlConfig)

cs.store(name="minecraft_3D_maze", group="task", node=MinecraftMazeConfig)
cs.store(name="minecraft_3D_holey_maze", group="task", node=MinecraftHoleyMazeConfig)
cs.store(name="minecraft_3D_dungeon_holey", group="task", node=MinecraftHoleyDungeonConfig)

cs.store(name="lego", group="task", node=LegoConfig)

cs.store(name="default_model", group="model", node=ModelConfig)
cs.store(name="seqnca", group="model", node=SeqNCAConfig)