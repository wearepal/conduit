from abc import abstractmethod
import importlib
import logging
import os
from pathlib import Path
import re
import sys
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import attr
import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.hydra import SchemaRegistration
from omegaconf import OmegaConf
import pytorch_lightning as pl
from typing_extensions import final

from conduit.data.datamodules.base import CdtDataModule
from conduit.models.base import CdtModel

__all__ = [
    "CdtRelay",
    "Relay",
    "SchemaInfo",
]


class SchemaInfo(NamedTuple):
    name: str
    conf: Type


R = TypeVar("R", bound="Relay")


@attr.define(kw_only=True)
class Relay:
    _CONFIG_NAME: ClassVar[str] = "config"
    _SCHEMA_NAME: ClassVar[str] = "experiment_schema"
    _CONFIGEN_FILENAME: ClassVar[str] = ".conf"
    _logger: ClassVar[Optional[logging.Logger]] = None

    @classmethod
    def _get_logger(cls: Type[R]) -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.INFO)
            cls._logger = logger
        return cls._logger

    @classmethod
    def log(cls: Type[R], msg: str) -> None:
        cls._get_logger().info(msg)

    @classmethod
    def _config_dir_name(cls: Type[R]) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    @final
    @classmethod
    def _init_config_dir(
        cls: Type[R], *, config_dir: Path, config_dict: Dict[str, Sequence[Any]]
    ) -> None:
        config_dir.mkdir(parents=True)
        cls.log(f"Initialising config directory '{config_dir}'")
        indent = "  "
        with open((config_dir / cls._CONFIG_NAME).with_suffix(".yaml"), "w") as exp_config:
            cls.log(f"Initialising primary config file '{exp_config.name}'")
            exp_config.write(f"defaults:")
            exp_config.write(f"\n{indent}- {cls._SCHEMA_NAME}")

            for group, schema_ls in config_dict.items():
                group_dir = config_dir / group
                group_dir.mkdir()
                cls.log(f"Initialising group '{group}'")
                for info in schema_ls:
                    open((group_dir / "defaults").with_suffix(".yaml"), "a").close()
                    with open((group_dir / info.name).with_suffix(".yaml"), "w") as schema_config:
                        schema_config.write(f"defaults:")
                        schema_config.write(f"\n{indent}- /schema/{group}: {info.name}")
                        schema_config.write(f"\n{indent}- defaults")
                        cls.log(f"- Initialising schema file '{schema_config.name}'")
                default = "null" if len(schema_ls) > 1 else schema_ls[0].name
                exp_config.write(f"\n{indent}- {group}: {default}")

        cls.log(f"Finished initialising config directory initialised at '{config_dir}'")

    @classmethod
    def _create_conf(cls: Type[R], config_dir: Path) -> None:
        from configen.config import ConfigenConf, ModuleConf  # type: ignore
        from configen.configen import generate_module, save  # type: ignore

        cfg = ConfigenConf(
            output_dir=str(config_dir),
            module_path_pattern="{{module_path}}" + f"/{cls._CONFIGEN_FILENAME}.py",
            modules=[],
            header="",
        )
        module_conf = ModuleConf(name=cls.__module__, classes=[cls.__name__])
        code = generate_module(cfg=cfg, module=module_conf)
        save(cfg=cfg, module=str(cfg.output_dir), code=code)

    @classmethod
    def _get_config_class(cls: Type[R], config_dir: Path) -> Any:
        conf_class_file = (config_dir / cls._CONFIGEN_FILENAME).with_suffix(".py")
        if not (conf_class_file).exists():
            cls._create_conf(config_dir=config_dir)
        spec = importlib.util.spec_from_file_location(  # type: ignore
            name=conf_class_file.name, location=str(conf_class_file)
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)
        sys.modules[conf_class_file.name] = module
        return getattr(module, f"{cls.__name__}Conf")

    @classmethod
    def with_hydra(
        cls: Type[R],
        base_config_dir: Union[Path, str],
        **schemas: SchemaInfo,
    ) -> None:
        cls._launch(base_config_dir=base_config_dir, **schemas)

    @final
    @classmethod
    def _launch(
        cls: Type[R], *, base_config_dir: Union[Path, str], **schemas: Sequence[SchemaInfo]
    ) -> None:
        base_config_dir = Path(base_config_dir)
        config_dir_name = cls._config_dir_name()
        config_dir = (base_config_dir / config_dir_name).expanduser().resolve()

        if not config_dir.exists():
            cls.log(
                f"Config directory {config_dir} not found."
                "\nInitialising directory based on the supplied conf classes."
            )
            cls._init_config_dir(config_dir=config_dir, config_dict=schemas)
            cls.log(f"Relaunch the relay, modifying the config files first if desired.")
            return

        config_class = cls._get_config_class(config_dir=config_dir)
        sr = SchemaRegistration()
        sr.register(path=cls._SCHEMA_NAME, config_class=config_class)
        for group, schema_ls in schemas.items():
            with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group:
                for info in schema_ls:
                    group.add_option(name=info.name, config_class=info.conf)

        # config_path only allows for relative paths; we need to resort to argv-manipulation
        # in order to set the config directory with an absolute path
        sys.argv.extend(["--config-dir", str(config_dir)])

        @hydra.main(config_path=None, config_name=cls._CONFIG_NAME)
        def launcher(cfg: Any) -> None:
            exp: R = instantiate(cfg, _recursive_=True)
            config_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, enum_to_str=True))
            exp.run(config_dict)

        launcher()

    @abstractmethod
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        ...


@attr.define(kw_only=True)
class CdtRelay(Relay):
    datamodule: CdtDataModule
    trainer: pl.Trainer
    model: CdtModel
    seed: Optional[int] = 42

    @classmethod
    def with_hydra(
        cls: Type[R],
        base_config_dir: Union[Path, str],
        *,
        datamodule_confs: List[SchemaInfo],
        model_confs: List[SchemaInfo],
    ) -> None:
        from conduit.hydra.pytorch_lightning.trainer.conf import (
            TrainerConf,  # type: ignore
        )

        configs = dict(
            datamodule=datamodule_confs,
            model=model_confs,
            trainer=[SchemaInfo(name="trainer", conf=TrainerConf)],
        )
        super().with_hydra(base_config_dir=base_config_dir, **configs)

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
