# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
import sys
import time


import itertools
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from hydra.types import HydraContext
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper, BasicSweeperConf
from hydra_plugins.hydra_submitit_launcher.config import LocalQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from hydra._internal.core_plugins.basic_launcher import BasicLauncher
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside sweep()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)


@dataclass
class CrossEvalLauncherConf(LocalQueueConf):
    _target_: str = "hydra_plugins.cross_eval_launcher_plugin.cross_eval_launcher.CrossEvalLauncher"


ConfigStore.instance().store(
    group="hydra/launcher", name="cross-eval", node=CrossEvalLauncherConf, provider="hydra"
)


# TODO: We can replicate this without a plugin?
class CrossEvalLauncher(BaseSubmititLauncher):
    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        setup_globals()
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None
        # configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        # sweep_dir = self.config.hydra.sweep.dir
        # Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
        # log.info(f"Cross-evaluating {len(job_overrides)} jobs locally")
        sweep_configs = []
        for idx, overrides in enumerate(job_overrides):
            # idx = initial_job_idx + idx
            # lst = " ".join(filter_overrides(overrides))
            # log.info(f"\t#{idx} : {lst}")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            # with open_dict(sweep_config):
            #     sweep_config.hydra.job.id = idx
            #     sweep_config.hydra.job.num = idx
            sweep_configs.append(sweep_config)

        sweep_params = self.config.hydra.sweeper.params

        # from cross_eval import cross_evaluate
        from control_pcgrl.rl.cross_eval import cross_evaluate
        cross_evaluate(self.config, sweep_configs, sweep_params)

        # Avoid unhappy hydra
        sys.exit()
