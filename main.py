"""Main script."""
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from real_patch.hydra_config.pytorch_lightning.trainer.conf import TrainerConf
from real_patch.hydra_config.real_patch.data.datamodules.conf import (
    CelebaDataModuleConf,
)
from real_patch.hydra_config.real_patch.finetune.conf import ResnetFtBaselineConf


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="experiment_schema", node=ResnetFtBaselineConf)
cs.store(
    group="schema/datamodule",
    name="celeba_schema",
    node=CelebaDataModuleConf,
    package="datamodule",
)

cs.store(group="schema/trainer", name="trainer_schema", node=TrainerConf, package="trainer")


@hydra.main(config_path="conf", config_name="ft_baseline")
def launcher(cfg: ResnetFtBaselineConf) -> None:
    cfg.datamodule.image_dir = to_absolute_path(cfg.datamodule.image_dir)
    cfg.datamodule.train_attr_path = to_absolute_path(cfg.datamodule.train_attr_path)
    cfg.datamodule.test_attr_path = to_absolute_path(cfg.datamodule.test_attr_path)
    exp = instantiate(cfg, _recursive_=True)
    exp.start(OmegaConf.to_container(cfg))


if __name__ == "__main__":
    launcher()
