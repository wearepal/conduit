from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import (
    BatchesProcessedColumn,
    CustomBarColumn,
    CustomTimeColumn,
    ProcessingSpeedColumn,
    RichProgressBar,
    RichProgressBarTheme,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich.console import RenderableType
from rich.progress import Task, TaskID, TextColumn
from rich.table import Column
from rich.text import Text
from typing_extensions import TypeAlias, override

__all__ = ["CdtProgressBar", "ProgressBarTheme"]


class _FixedLengthProcessionSpeed(ProcessingSpeedColumn):
    """Renders processing speed for the progress bar with fixes length"""

    def __init__(self, style: Union[str, Any]) -> None:
        super().__init__(style)
        self.max_length = len("0.00")

    def render(self, task: Task) -> RenderableType:
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        self.max_length = max(len(task_speed), self.max_length)
        task_speed = " " * (self.max_length - len(task_speed)) + task_speed
        return Text(f"{task_speed}it/s", style=self.style, justify="center")


def _quaterion_theme() -> RichProgressBarTheme:
    return RichProgressBarTheme(
        description="white",
        progress_bar="#4881AD",
        progress_bar_finished="#67C87A",
        progress_bar_pulse="#67C87A",
        batch_progress="white",
        time="grey54",
        processing_speed="grey70",
        metrics="white",
    )


def _cyberpunk_theme() -> RichProgressBarTheme:
    return RichProgressBarTheme(
        description="#FF00CF",
        progress_bar="#001eff",
        progress_bar_finished="#00ff9f",
        progress_bar_pulse="#001eff",
        batch_progress="#FF00CF",
        time="#00b8ff",
        processing_speed="#FF46D6",
        metrics="#00ff9f",
    )


def _google_theme() -> RichProgressBarTheme:
    return RichProgressBarTheme(
        description="#008744",
        progress_bar="#0057e7",
        progress_bar_finished="#d62d20",
        progress_bar_pulse="#d62d20",
        batch_progress="#ffa700",
        time="white",
        processing_speed="#0057e7",
        metrics="#d62d20",
    )


class ProgressBarTheme(Enum):
    QUATERION = (_quaterion_theme,)
    CYBERPUNK = (_cyberpunk_theme,)
    GOOGLE = (_google_theme,)

    def __init__(self, load: Callable[..., RichProgressBarTheme]) -> None:
        self.load = load


class CdtProgressBar(RichProgressBar):
    """Custom Lightning progress bar supporting both epoch- and iteration-based training."""

    Theme: TypeAlias = ProgressBarTheme

    def __init__(
        self,
        refresh_rate: int = 1,
        *,
        leave: bool = False,
        theme: Union[RichProgressBarTheme, Theme] = Theme.CYBERPUNK,
        console_kwargs: Optional[Dict[str, Any]] = None,
        predict_description: str = "Predicting",
    ) -> None:
        if isinstance(theme, ProgressBarTheme):
            theme = cast(RichProgressBarTheme, theme.load())

        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme,
            console_kwargs=console_kwargs,
        )
        self.predict_progress_bar_id = None
        self._predict_description = predict_description

    @override
    def configure_columns(self, *args: Any, **kwargs: Any) -> list:  # pyright: ignore
        return [
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(
                    no_wrap=True,
                    min_width=9,  # prevents blinking during validation, length of `Validation `
                ),
            ),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            _FixedLengthProcessionSpeed(style=self.theme.processing_speed),
        ]

    @staticmethod
    def _is_iteration_based(trainer: pl.Trainer) -> bool:
        return trainer.num_training_batches == float("inf")

    @property
    @override
    def total_train_batches(self) -> Union[int, float]:
        """
        The total number of training batches, which may change from epoch to epoch.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if
        the training dataloader is of infinite size.
        """
        if self.trainer.max_steps >= 0:
            return self.trainer.max_steps
        return self.trainer.num_training_batches

    @override
    def _add_task(
        self, total_batches: float, description: str, visible: bool = True
    ) -> Optional[TaskID]:
        if self.progress is not None:
            return self.progress.add_task(
                f"[{self.theme.description}]{description}", total=total_batches, visible=visible
            )

    @override
    def _update(self, progress_bar_id: int, current: int, visible: bool = True) -> None:
        if self.progress is not None and self.is_enabled:
            total = self.progress.tasks[progress_bar_id].total
            if not self._should_update(current, total):  # type: ignore
                return

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(TaskID(progress_bar_id), advance=advance, visible=visible)
            self.refresh()

    @override
    def _get_train_description(self, current_epoch: Optional[int]) -> str:
        train_description = f"Training"
        if current_epoch is not None:
            train_description += f" (Epoch: {current_epoch})"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            num_digits = len(str(current_epoch))
            required_padding = (
                len(self.validation_description) - len(train_description) + 1
            ) - num_digits
            for _ in range(required_padding):
                train_description += " "
        return train_description

    @property
    def _progress_bar_id(self) -> Optional[TaskID]:
        return getattr(self, "train_progress_bar_id", getattr(self, "main_progress_bar_id", None))

    @_progress_bar_id.setter
    def _progress_bar_id(self, value: Optional[TaskID]) -> None:
        if hasattr(self, "train_progress_bar_id"):
            self.train_progress_bar_id = value
        if hasattr(self, "main_progress_bar_id"):
            self.main_progress_bar_id = value

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:  # pyright: ignore
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch

        iteration_based = self._is_iteration_based(trainer)
        current_epoch = None if iteration_based else trainer.current_epoch
        train_description = self._get_train_description(current_epoch)
        if self._progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self._progress_bar_id is None:
            self._progress_bar_id = self._add_task(total_train_batches, train_description)
        elif (not iteration_based) and (self.progress is not None):
            self.progress.reset(
                TaskID(self._progress_bar_id),
                total=total_train_batches,
                description=train_description,
                visible=True,
            )
        self.refresh()

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # pyright: ignore
        outputs: Optional[STEP_OUTPUT],  # pyright: ignore
        batch: Any,  # pyright: ignore
        batch_idx: int,
        dataloader_idx: int = 0,  # pyright: ignore
    ) -> None:
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, batch_idx + 1)  # type: ignore
        elif self.val_progress_bar_id is not None:
            # check to see if we should update the main training progress bar
            self._update(self.val_progress_bar_id, batch_idx + 1)
        self.refresh()

    @property
    def predict_description(self) -> str:
        return self._predict_description

    @predict_description.setter
    def predict_description(self, value: str) -> None:
        self._predict_description = value

    @override
    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,  # pyright: ignore
        pl_module: pl.LightningModule,  # pyright: ignore
        batch: Any,  # pyright: ignore
        batch_idx: int,  # pyright: ignore
        dataloader_idx: int = 0,
    ) -> None:
        if self.has_dataloader_changed(dataloader_idx):
            if (self.predict_progress_bar_id is not None) and (self.progress is not None):
                self.progress.update(TaskID(self.predict_progress_bar_id), advance=0, visible=False)
            self.predict_progress_bar_id = self._add_task(
                self.total_predict_batches_current_dataloader, self.predict_description
            )
            self.refresh()
