from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from tokenizer.engine.module import TrainableModule


class Trainer:
    def __init__(
        self,
        max_epochs: int = 100,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_every_n_steps: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.max_epochs = max_epochs
        self.checkpoint_dir: Union[None, Path] = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.checkpoint_every_n_steps: Union[None, int] = checkpoint_every_n_steps
        self.device = self._setup_device(device)

    def _setup_device(self, device: Union[str, None]) -> str:
        if (device in ("cuda", None)) and torch.cuda.is_available():
            return "cuda"
        if device != "cpu":
            raise Warning(
                "Unknown device specified. Falling back to torch CPU backend."
            )
        return "cpu"

    def fit(
        self,
        module: TrainableModule,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
        checkpoint_prefix: str = "",
    ):
        module = module.to(self.device)

        if isinstance(self.checkpoint_dir, Path):
            # TODO: what about checkpoint_dir = None?
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(module.epoch, self.max_epochs):
            module.epoch = epoch
            module.train()
            module.on_epoch_start()
            for batch_idx, batch in tqdm(
                enumerate(train_dataloader), desc=f"Train epoch {epoch}"
            ):
                batch = self._move_batch_to_device(batch)
                module.training_step(batch, batch_idx)
                if (
                    self.checkpoint_every_n_steps
                    and (module.step % self.checkpoint_every_n_steps) == 0
                ):
                    module.save_checkpoint(
                        file_path=self.checkpoint_dir
                        / f"{checkpoint_prefix}_step={module.step}.ckpt"
                    )
            module.on_epoch_end()
            if valid_dataloader:
                module.eval()
                for batch_idx, batch in tqdm(
                    enumerate(valid_dataloader), desc=f"Validation epoch {epoch}"
                ):
                    batch = self._move_batch_to_device(batch)
                    module.validation_step(batch, batch_idx)
            if self.checkpoint_every_n_steps is None:
                module.save_checkpoint(
                    file_path=self.checkpoint_dir
                    / f"{checkpoint_prefix}_step={module.step}.ckpt"
                )

    def _move_batch_to_device(
        self, batch: Union[torch.Tensor, tuple, dict]
    ) -> Union[torch.Tensor, tuple, dict]:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (tuple, list)):
            return tuple(
                t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch
            )
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        raise ValueError(
            f"Dataloader output must be either a tensor, a tuple or a dict."
        )
