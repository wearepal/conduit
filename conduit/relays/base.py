from abc import abstractmethod
from collections import defaultdict
from dataclasses import is_dataclass
from functools import lru_cache
import importlib
import logging
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import attr
import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.decorators import implements
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


@lru_cache()
def _camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


@attr.define(frozen=True)
class SchemaInfo:
    conf: Type[Any]
    _name: Optional[str] = None

    @property
    def name(self) -> str:
        if self._name is None:
            cls_name = self.conf.__name__
            if cls_name.endswith("Conf"):
                cls_name.rstrip("Conf")
            return _camel_to_snake(cls_name)
        return self._name


@attr.define(frozen=True)
class _SchemaImportInfo:
    conf_name: str
    name: str


R = TypeVar("R", bound="Relay")


@attr.define(kw_only=True)
class Relay:
    """
    Abstract class for orchestrating hydra runs.

    This class does away with the hassle of needing to define config-stores, initialise
    config directories, and manually run configen on classes to convert them into schemas.
    Regular non-hydra compatible, classes can be passed to the `with_hydra` method and
    configen will be run on them automatically, with the resulting conf classes being
    cached in the config directory.

    Subclasses must implement a 'run' method.

    >>>
    Relay.with_hydra(
        base_config_dir="conf",
        model_confs=[SchemaInfo(MoCoV2), SchemaInfo(DINO)],
        datamodule_confs=[SchemaInfo(ColoredMNISTDataModule, "cmnist")],
    )

    """

    _CONFIG_NAME: ClassVar[str] = "config"
    _PRIMARY_SCHEMA_NAME: ClassVar[str] = "relay_schema"
    _CONFIGEN_FILENAME: ClassVar[str] = "conf.py"
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
        return _camel_to_snake(cls.__name__)

    @final
    @classmethod
    def _init_yaml_files(
        cls: Type[R], *, config_dir: Path, config_dict: Dict[str, List[Any]]
    ) -> None:
        indent = "  "
        primary_conf_fp = (config_dir / cls._CONFIG_NAME).with_suffix(".yaml")
        primary_conf_exists = primary_conf_fp.exists()
        with primary_conf_fp.open("a+") as primary_conf:
            if not primary_conf_exists:
                cls.log(f"Initialising primary config file '{primary_conf.name}'.")

                primary_conf.write(f"defaults:")
                primary_conf.write(f"\n{indent}- {cls._PRIMARY_SCHEMA_NAME}")

            for group, schema_ls in config_dict.items():
                group_dir = config_dir / group
                if not group_dir.exists():
                    group_dir.mkdir()
                    default = "null" if len(schema_ls) > 1 else schema_ls[0].name
                    primary_conf.write(f"\n{indent}- {group}: {default}")

                cls.log(f"Initialising group '{group}'")
                for info in schema_ls:
                    open((group_dir / "defaults").with_suffix(".yaml"), "a").close()
                    with (group_dir / info.name).with_suffix(".yaml").open("w") as schema_config:
                        schema_config.write(f"defaults:")
                        schema_config.write(f"\n{indent}- /schema/{group}: {info.name}")
                        schema_config.write(f"\n{indent}- defaults")
                        cls.log(f"- Initialising config file '{schema_config.name}'.")

        cls.log(f"Finished initialising config directory initialised at '{config_dir}'")

    @classmethod
    def _generate_conf(
        cls: Type[R], output_dir: Path, *, module_class_dict: Dict[str, List[str]]
    ) -> None:
        from configen.config import ConfigenConf, ModuleConf  # type: ignore
        from configen.configen import generate_module  # type: ignore

        cfg = ConfigenConf(
            output_dir=str(output_dir),
            module_path_pattern=f"{cls._CONFIGEN_FILENAME}",
            modules=[],
            header="",
        )
        for module, classes in module_class_dict.items():
            module_conf = ModuleConf(name=module, classes=classes)
            code = generate_module(cfg=cfg, module=module_conf)
            output_dir.mkdir(parents=True, exist_ok=True)
            conf_file = output_dir / cls._CONFIGEN_FILENAME
            with conf_file.open("a+") as file:
                file.write(code)

    @classmethod
    def _load_module_from_path(cls: Type[R], filepath: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(  # type: ignore
            name=filepath.name, location=str(filepath)
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)
        sys.modules[filepath.name] = module
        return module

    @classmethod
    def _load_schemas(
        cls: Type[R],
        config_dir: Path,
        *,
        use_cached_confs: bool = True,
        **options: Sequence[SchemaInfo],
    ) -> Tuple[Type[Any], DefaultDict[str, List[SchemaInfo]], DefaultDict[str, List[SchemaInfo]]]:
        configen_dir = config_dir / "configen"
        schema_filepath = configen_dir / cls._CONFIGEN_FILENAME
        schemas_to_generate = defaultdict(list)
        if not use_cached_confs:
            schema_filepath.unlink(missing_ok=True)  # type: ignore
        if schema_filepath.exists():
            module = cls._load_module_from_path(schema_filepath)
        else:
            schemas_to_generate[cls.__module__].append(cls.__name__)
            module = None
        imported_schemas: DefaultDict[str, List[SchemaInfo]] = defaultdict(list)
        schemas_to_import: DefaultDict[str, List[_SchemaImportInfo]] = defaultdict(list)
        schemas_to_init: DefaultDict[str, List[SchemaInfo]] = defaultdict(list)

        for group, classes in options.items():
            for info in classes:
                if not (config_dir / group / info.name).with_suffix(".yaml").exists():
                    schemas_to_init[group].append(info)
                cls_name = info.conf.__name__
                if (not is_dataclass(info.conf)) or (not cls_name.endswith("Conf")):
                    schema_name = f"{cls_name}Conf"
                    schema_missing = False
                    if module is None:
                        schema_missing = True
                    else:
                        schema = getattr(module, schema_name, None)
                        if schema is None:
                            schema_missing = True
                        else:
                            imported_schemas[group].append(
                                SchemaInfo(name=info.name, conf=schema)  # type: ignore
                            )
                    if schema_missing:
                        schemas_to_generate[info.conf.__module__].append(cls_name)
                    import_info = _SchemaImportInfo(conf_name=schema_name, name=info.name)
                    schemas_to_import[group].append(import_info)
                else:
                    imported_schemas[group].append(info)

        # Generate any confs with configen that have yet to be generated
        if schemas_to_generate:
            cls._generate_conf(output_dir=configen_dir, module_class_dict=schemas_to_generate)
        # Load the primary schema
        module = cls._load_module_from_path(schema_filepath)
        primary_schema = getattr(module, cls.__name__ + "Conf")
        # Load the sub-schemas
        for group, info_ls in schemas_to_import.items():
            for info in info_ls:
                imported_schemas[group].append(
                    SchemaInfo(name=info.name, conf=getattr(module, info.conf_name))  # type: ignore
                )

        return primary_schema, imported_schemas, schemas_to_init

    @classmethod
    def with_hydra(
        cls: Type,
        base_config_dir: Union[Path, str],
        use_cached_confs: bool = True,
        **schemas: SchemaInfo,
    ) -> None:
        cls._launch(base_config_dir=base_config_dir, use_cached_confs=use_cached_confs, **schemas)

    @final
    @classmethod
    def _launch(
        cls: Type[R],
        *,
        base_config_dir: Union[Path, str],
        use_cached_confs: bool = True,
        **options: Sequence[SchemaInfo],
    ) -> None:
        base_config_dir = Path(base_config_dir)
        config_dir_name = cls._config_dir_name()
        config_dir = (base_config_dir / config_dir_name).expanduser().resolve()
        config_dir.mkdir(exist_ok=True, parents=True)

        primary_schema, schemas, schemas_to_init = cls._load_schemas(
            config_dir=config_dir, use_cached_confs=use_cached_confs, **options
        )
        # Initialise any missing yaml files
        if schemas_to_init:
            cls.log(
                f"One or more config files not found in config directory {config_dir}."
                "\nInitialising missing config files."
            )
            cls._init_yaml_files(config_dir=config_dir, config_dict=schemas_to_init)
            cls.log(f"Relaunch the relay, modifying the config files first if desired.")
            return

        sr = SchemaRegistration()
        sr.register(path=cls._PRIMARY_SCHEMA_NAME, config_class=primary_schema)
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
        use_cached_confs: bool = True,
        datamodule_confs: List[SchemaInfo],
        model_confs: List[SchemaInfo],
    ) -> None:
        from conduit.hydra.pytorch_lightning.trainer.conf import (
            TrainerConf,  # type: ignore
        )

        configs = dict(
            datamodule=datamodule_confs,
            model=model_confs,
            trainer=[SchemaInfo(name="trainer", conf=TrainerConf)],  # type: ignore
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
