import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import attr
from hydra.utils import to_absolute_path
import pytorch_lightning as pl
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

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
        root: Union[Path, str],
        *,
        datamodule: List[Union[Type[Any], Option]],
        model: List[Union[Type[Any], Option]],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            datamodule=datamodule,
            model=model,
            trainer=[Option(class_=pl.Trainer, name="trainer")],
        )
        super().with_hydra(root=root, clear_cache=clear_cache, **configs)

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
            self.datamodule.root = to_absolute_path(self.datamodule.root)  # type: ignore
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)
