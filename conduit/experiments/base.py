import os
from pathlib import Path
import re
from typing import Any, Dict, List, NamedTuple, Optional, Type, TypeVar, Union

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

__all__ = [
    "CdtExperiment",
    "SchemaInfo",
]


class SchemaInfo(NamedTuple):
    name: str
    conf: Type


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
    def _config_name(cls: Type[E]) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    @classmethod
    def launch(
        cls: Type[E], *, datamodule_confs: List[SchemaInfo], model_confs: List[SchemaInfo]
    ) -> None:
        app = typer.Typer()

        @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
        def launcher(ctx: typer.Context, config_dir: Path = typer.Option(Path("conf"))) -> None:
            conf_cls_name = f"{cls.__name__}Conf"

            import conduit.hydra.conduit.experiments.conf as exp_confs

            try:
                conf_cls = getattr(exp_confs, conf_cls_name)
            except AttributeError:
                raise AttributeError(
                    f"Config class for {cls.__name__} could not be found in {str(exp_confs)}."
                    "Please try generating it with configen before trying again."
                )

            from conduit.hydra.pytorch_lightning.trainer.conf import TrainerConf

            conf_dict = dict(
                datamodule=datamodule_confs,
                model=model_confs,
                trainer=[SchemaInfo(name="trainer", conf=TrainerConf)],
            )

            sr = SchemaRegistration()
            sr.register(path="experiment_schema", config_class=conf_cls)
            for group, schema_ls in conf_dict.items():
                with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group:
                    for name, conf in schema_ls:
                        group.add_option(name=name, config_class=conf)

            with hydra.initialize_config_dir(
                config_dir=str(config_dir.expanduser().resolve()), job_name="run"
            ):
                cfg = hydra.compose(config_name=cls._config_name(), overrides=ctx.args)
                print(f"Current working directory: f{os.getcwd()}")
                if hasattr(cfg.datamodule, "root"):
                    cfg.datamodule.root = to_absolute_path(cfg.datamodule.root)  # type: ignore
                exp: E = instantiate(cfg, _recursive_=True)
                exp.run(OmegaConf.to_container(cfg, enum_to_str=True))

        app()
