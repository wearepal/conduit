from abc import abstractmethod
from collections import defaultdict
from dataclasses import is_dataclass
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
    Dict,
    List,
    NamedTuple,
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
    conf: Type[Any]


class _SchemaImportInfo(NamedTuple):
    name: str
    class_name: str
    module: ModuleType


R = TypeVar("R", bound="Relay")


@attr.define(kw_only=True)
class Relay:
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
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    @final
    @classmethod
    def _init_config_dir(
        cls: Type[R], *, config_dir: Path, config_dict: Dict[str, Sequence[Any]]
    ) -> None:
        # TODO: Make work for subsequent runs when new options are added
        config_dir.mkdir(parents=True)
        cls.log(f"Initialising config directory '{config_dir}'")
        indent = "  "
        with open((config_dir / cls._CONFIG_NAME).with_suffix(".yaml"), "w") as exp_config:
            cls.log(f"Initialising primary config file '{exp_config.name}'")
            exp_config.write(f"defaults:")
            exp_config.write(f"\n{indent}- {cls._PRIMARY_SCHEMA_NAME}")

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
    def _generate_conf(
        cls: Type[R], output_dir: Path, *, module_class_dict: Dict[str, List[str]]
    ) -> None:
        from configen.config import ConfigenConf, ModuleConf  # type: ignore
        from configen.configen import generate_module, save  # type: ignore

        cfg = ConfigenConf(
            output_dir=str(output_dir),
            module_path_pattern="{{module_path}}" + f"/{cls._CONFIGEN_FILENAME}",
            modules=[],
            header="",
        )
        for module, classes in module_class_dict.items():
            module_conf = ModuleConf(name=module, classes=classes)
            code = generate_module(cfg=cfg, module=module_conf)
            # TODO: Change saving to append if file already exists
            save(cfg=cfg, module=module, code=code)

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
    def _get_schemas(
        cls: Type[R], config_dir: Path, **options: Sequence[SchemaInfo]
    ) -> Tuple[Type[Any], Dict[str, List[SchemaInfo]]]:
        configen_dir = config_dir / "configen"
        module_path_ps = configen_dir / cls.__module__.replace(".", "/") / cls._CONFIGEN_FILENAME
        to_generate = defaultdict(list)
        if not module_path_ps.exists():
            to_generate[cls.__module__].append(cls.__name__)

        imported_schemas = defaultdict(list)
        to_import = defaultdict(list)

        for group, classes in options.items():
            for info in classes:
                module_path_ss = (
                    configen_dir / info.conf.__module__.replace(".", "/") / cls._CONFIGEN_FILENAME
                )
                module = (
                    cls._load_module_from_path(module_path_ss) if module_path_ss.exists() else None
                )

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
                            imported_schemas[group].append(SchemaInfo(name=info.name, conf=schema))
                    if schema_missing:
                        to_generate[info.conf.__module__].append(cls_name)
                    import_info = _SchemaImportInfo(
                        name=info.name, class_name=schema_name, module=module_path_ss
                    )
                    to_import[group].append(import_info)
                else:
                    imported_schemas[group].append(info)

        # Generate any confs with configen that have yet to be generated
        if to_generate:
            cls._generate_conf(output_dir=configen_dir, module_class_dict=to_generate)

        # Load the primary schema
        module = cls._load_module_from_path(module_path_ps)
        primary_schema = getattr(module, cls.__name__ + "Conf")
        # Load the sub-schemas
        for group, names in to_import.items():
            for name, schema_cls_name, module_path in names:
                module = cls._load_module_from_path(module_path)
                imported_schemas[group].append(
                    SchemaInfo(name=name, conf=getattr(module, schema_cls_name))
                )

        return primary_schema, imported_schemas

    @classmethod
    def with_hydra(
        cls: Type,
        base_config_dir: Union[Path, str],
        **schemas: SchemaInfo,
    ) -> None:
        cls._launch(base_config_dir=base_config_dir, **schemas)

    @final
    @classmethod
    def _launch(
        cls: Type[R], *, base_config_dir: Union[Path, str], **options: Sequence[SchemaInfo]
    ) -> None:
        base_config_dir = Path(base_config_dir)
        config_dir_name = cls._config_dir_name()
        config_dir = (base_config_dir / config_dir_name).expanduser().resolve()

        if not config_dir.exists():
            cls.log(
                f"Config directory {config_dir} not found."
                "\nInitialising directory based on the supplied conf classes."
            )
            cls._init_config_dir(config_dir=config_dir, config_dict=options)
            cls.log(f"Relaunch the relay, modifying the config files first if desired.")
            return

        primary_schema, schemas = cls._get_schemas(config_dir=config_dir, **options)
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
