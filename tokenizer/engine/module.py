from typing import Union, Tuple, Dict, Sequence, Optional, final
from collections.abc import Callable
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class TrainableModule(nn.Module, ABC):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, "_post_init_hook"):
                self._post_init_hook()

        cls.__init__ = wrapped_init

    def _post_init_hook(self):
        """Executes after the constructor of the derived class."""
        if not hasattr(self, "hparams"):
            self.hparams: dict = {}
        self.optimizers, self.lr_schedulers = self.configure_optimizers()
        self.epoch: int = 0  # Does this belong here?
        self.step: int = 0
        self.logger: Union[SummaryWriter, None] = None

    def set_logger(self, logger: SummaryWriter):
        self.logger = logger

    def log_scalar(self, tag: str, scalar: Union[float, int], step: int, **kwargs):
        if self.logger:
            self.logger.add_scalar(tag, scalar, step, **kwargs)

    @final
    def training_step(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
        batch_idx: int,
    ) -> Union[None, torch.Tensor]:
        self._training_step(batch, batch_idx)
        self.step += 1

    @abstractmethod
    def _training_step(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
        batch_idx: int,
    ) -> Union[None, torch.Tensor]:
        pass

    @final
    def validation_step(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
        batch_idx: int,
    ) -> None:
        self._validation_step(batch, batch_idx)

    def _validation_step(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
        batch_idx: int,
    ) -> None:
        pass

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> Tuple[
        Sequence[torch.optim.Optimizer], Sequence[torch.optim.lr_scheduler.LRScheduler]
    ]:
        pass

    def save_checkpoint(self, file_path: Union[str, Path]):
        checkpoint = {
            "model_state": self.state_dict(),
            "hparams": self.hparams,
            "optimizer_states": [opt.state_dict() for opt in self.optimizers],
            "scheduler_states": [sch.state_dict() for sch in self.lr_schedulers],
            "epoch": self.epoch,
            "step": self.step,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        map_location: Optional[str] = None,
        **extra_kwargs,
    ) -> "TrainableModule":
        checkpoint = torch.load(file_path, map_location=map_location)
        hparams = checkpoint["hparams"]
        module = cls(**hparams, **extra_kwargs)
        module.load_state_dict(checkpoint["model_state"])

        if (opt_len := len(module.optimizers)) != (
            opt_state_len := len(checkpoint["optimizer_states"])
        ):
            raise ValueError(
                f"Optimizer count mismatch: Module has {opt_len} optimizer(s), "
                f"but checkpoint contains {opt_state_len} state(s)."
            )
        for opt, opt_state in zip(module.optimizers, checkpoint["optimizer_states"]):
            opt.load_state_dict(opt_state)

        if (sched_len := len(module.lr_schedulers)) != (
            sched_state_len := len(checkpoint["scheduler_states"])
        ):
            raise ValueError(
                f"LRScheduler count mismatch: Module has {sched_len} schedulers(s), "
                f"but checkpoint contains {sched_state_len} state(s)."
            )
        for sched, sched_state in zip(
            module.lr_schedulers, checkpoint["scheduler_states"]
        ):
            sched.load_state_dict(sched_state)

        module.epoch = checkpoint.get("epoch", 0)
        module.step = checkpoint.get("step", 0)

        return module

    @staticmethod
    def save_hyperparameters(init_fn: Callable):

        def is_basic_type(obj) -> bool:
            if isinstance(obj, (int, float, bool, str, type(None))):
                return True
            if isinstance(obj, (list, tuple)):
                return all(is_basic_type(item) for item in obj)
            if isinstance(obj, dict):
                return all(
                    isinstance(k, str) and is_basic_type(v) for k, v in obj.items()
                )
            # TODO: support for basic type dataclasses (serializable to dict) ?
            return False

        def wrapper(self, *args, **kwargs):
            import inspect

            sig = inspect.signature(init_fn)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.hparams = {
                k: v
                for k, v in bound.arguments.items()
                if k != "self" and is_basic_type(v)
            }
            return init_fn(self, *args, **kwargs)

        return wrapper

    def print_parameter_overview(self):
        print(f"{'Module Name':<30} {'Module Type':<25} {'Trainable Params':>15}")
        print("-" * 75)
        for name, module in self.named_modules():
            if name == "":
                # Skip the root module itself
                continue
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<30} {type(module).__name__:<25} {param_count:>15,}")
        print("-" * 75)

    # --- callbacks ---

    def on_epoch_start(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        pass
