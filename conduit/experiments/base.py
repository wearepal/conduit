import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Type, TypeVar, Union

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

    _CONFIG_NAME: ClassVar[str] = "config"
    _SCHEMA_NAME: ClassVar[str] = "experiment_schema"
    _logger: ClassVar[Optional[logging.Logger]] = logging.getLogger(__file__)

    @classmethod
    def _get_logger(cls: Type[E]) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger(cls.__name__)
        return cls._logger

    @classmethod
    def log(cls: Type[E], msg: str) -> None:
        cls._get_logger().info(msg)

    def run(self, raw_config: Optional[Union[Dict[DictKeyType, Any], List[Any], str]] = None):
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.log(f"Current working directory: '{os.getcwd()}'")
        if raw_config is not None:
            self.log("-----\n" + str(raw_config) + "\n-----")
        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)

    @classmethod
    def _config_dir_name(cls: Type[E]) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    @classmethod
    def _init_dir(cls: Type[E], config_dir: Path, *, config_dict: Dict[str, List[Any]]) -> None:
        config_dir.mkdir(parents=True)
        cls.log(f"\nInitialising config directory '{config_dir}'")
        indent = "  "
        with open((config_dir / cls._CONFIG_NAME).with_suffix(".yaml"), "w") as exp_config:
            cls.log(f"\nInitialising primary config file '{exp_config.name}'")
            exp_config.write(f"defaults:")
            exp_config.write(f"\n{indent}- {cls._SCHEMA_NAME}")

            for group, schema_ls in config_dict.items():
                group_dir = config_dir / group
                group_dir.mkdir()
                cls.log(f"\nInitialising group '{group}'")
                for info in schema_ls:
                    open((group_dir / "defaults").with_suffix(".yaml"), "a").close()
                    with open((group_dir / info.name).with_suffix(".yaml"), "w") as schema_config:
                        schema_config.write(f"defaults:")
                        schema_config.write(f"\n{indent}- /schema/{group}: {info.name}")
                        schema_config.write(f"\n{indent}- defaults")
                        cls.log(f"- Initialising schema file '{schema_config.name}'")
                default = "null" if len(schema_ls) > 1 else schema_ls[0].name
                exp_config.write(f"\n{indent}- {group}: {default}")

        cls.log(f"\nFinished initialising config directory initialised at '{config_dir}'")

    @classmethod
    def launch(
        cls: Type[E],
        base_config_dir: Union[Path, str],
        *,
        datamodule_confs: List[SchemaInfo],
        model_confs: List[SchemaInfo],
    ) -> None:
        base_config_dir = Path(base_config_dir)
        config_dir_name = cls._config_dir_name()
        config_dir = (base_config_dir / config_dir_name).expanduser().resolve()

        import conduit.hydra.conduit.experiments.conf as exp_confs

        try:
            conf_cls = getattr(exp_confs, f"{cls.__name__}Conf")
        except AttributeError:
            raise AttributeError(
                f"Config class for {cls.__name__} could not be found in {exp_confs.__name__}."
                "Please generate it with configen before trying again."
            )

        from conduit.hydra.pytorch_lightning.trainer.conf import TrainerConf

        conf_dict = dict(
            datamodule=datamodule_confs,
            model=model_confs,
            trainer=[SchemaInfo(name="trainer", conf=TrainerConf)],
        )

        if not config_dir.exists():
            cls.log(
                f"Configuration directory {config_dir} not found."
                "\nInitialising directory based on the supplied conf classes."
            )
            cls._init_dir(config_dir=config_dir, config_dict=conf_dict)
            cls.log(f"Relaunch the experiment, modifying the config files first if desired.")
            return

        sr = SchemaRegistration()
        sr.register(path=cls._SCHEMA_NAME, config_class=conf_cls)
        for group, schema_ls in conf_dict.items():
            with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group:
                for info in schema_ls:
                    group.add_option(name=info.name, config_class=info.conf)

        # config_path only allows for relative paths; we need to resort to argv-manipulation
        # in order to set the config directory with an absolute path
        sys.argv.extend(["--config-dir", str(config_dir)])

        @hydra.main(config_path=None, config_name=cls._CONFIG_NAME)
        def launcher(cfg: cls) -> None:
            cls.log(f"Current working directory: f{os.getcwd()}")
            if hasattr(cfg.datamodule, "root"):
                cfg.datamodule.root = to_absolute_path(cfg.datamodule.root)  # type: ignore
            exp: E = instantiate(cfg, _recursive_=True)
            exp.run(OmegaConf.to_container(cfg, enum_to_str=True))

        launcher()
