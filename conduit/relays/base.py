import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import attr
from hydra.utils import to_absolute_path
from kit.decorators import implements
from kit.hydra import Option, Relay
import pytorch_lightning as pl

from conduit.data.datamodules.base import CdtDataModule
from conduit.models.base import CdtModel

__all__ = [
    "CdtRelay",
]


@attr.define(kw_only=True)
class CdtRelay(Relay):
    datamodule: CdtDataModule
    trainer: pl.Trainer
    model: CdtModel
    seed: Optional[int] = 42

    @classmethod
    def with_hydra(
        cls,
        base_config_dir: Union[Path, str],
        *,
        datamodules: List[Union[Type[Any], Option]],
        models: List[Union[Type[Any], Option]],
        use_cached_confs: bool = True,
    ) -> None:

        configs = dict(
            datamodule=datamodules,
            model=models,
            trainer=[Option(class_=pl.Trainer, name="trainer")],
        )
        super().with_hydra(
            base_config_dir=base_config_dir, use_cached_confs=use_cached_confs, **configs
        )

    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        if raw_config is not None:
            self.log("-----\n" + str(raw_config) + "\n-----")
            try:
                self.trainer.logger.log_hyperparams(config_dict)  # type: ignore
            except:
                ...

        if hasattr(self.datamodule, "root"):
            self.datamodule.root = to_absolute_path(cfg.datamodule.root)  # type: ignore
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)
