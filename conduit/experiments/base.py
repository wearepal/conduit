import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import attr
import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.hydra import SchemaRegistration
from omegaconf import OmegaConf
from omegaconf.base import DictKeyType
import pytorch_lightning as pl
import typer

from conduit.data.datamodules.base import CdtDataModule
from conduit.models.base import CdtModel

__all__ = ["CdtExperiment"]


E = TypeVar("E", bound="CdtExperiment")


@attr.define(kw_only=True)
class CdtExperiment:
    datamodule: CdtDataModule
    trainer: pl.Trainer
    model: CdtModel
    seed: Optional[int] = 42

    def run(self, raw_config: Optional[Union[Dict[DictKeyType, Any], List[Any], str]] = None):
        self.datamodule.prepare_data()
        self.datamodule.setup()
        print(f"Current working directory: '{os.getcwd()}'")
        if raw_config is not None:
            print("-----\n" + str(raw_config) + "\n-----")
        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)

    @classmethod
    def launch(cls: Type[E], *, datamodule_confs: List[Type], model_confs: List[Type]) -> None:
        app = typer.Typer()
        conf_dict = dict(datamodule=datamodule_confs, model=model_confs)

        @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
        def make_launcher(config_path: Path = Path("conf")) -> None:
            sr = SchemaRegistration()
            conf_cls_name = f"{cls.__name__}Conf"

            import conduit.hydra.conduit.experiments.conf as exp_confs

            try:

                conf_cls = getattr(exp_confs, conf_cls_name)
                breakpoint()
            except AttributeError:
                raise AttributeError(
                    f"Config class for {cls.__name__} could not be found in {str(exp_confs)}."
                    "Please try generating it with configen before trying again."
                )
            sr.register(path="experiment_schema", config_class=conf_cls)
            # Define the 'datamodule' group
            for group, conf_ls in conf_dict.items():
                with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group:
                    for conf in conf_ls:
                        name = conf.__name__
                        if name.endswith("Conf"):
                            name.rstrip("Conf")
                        group.add_option(name=name, config_class=conf)

            @hydra.main(config_path=str(config_path), config_name="ssl_experiment")
            def launcher(cfg: E) -> None:
                print(f"Current working directory: f{os.getcwd()}")
                if hasattr(cfg.datamodule, "root"):
                    cfg.datamodule.root = to_absolute_path(cfg.datamodule.root)  # type: ignore
                exp: E = instantiate(cfg, _recursive_=True)
                exp.run(OmegaConf.to_container(cfg, enum_to_str=True))

            del sys.argv[1]
            launcher()

        make_launcher()
