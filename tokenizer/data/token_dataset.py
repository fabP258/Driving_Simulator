from typing import Union
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


class VideoTokenDataset(Dataset):

    def __init__(
        self,
        token_root_path: Union[str, Path],
        num_cond_frames: int,
        H: int,
        W: int,
        max_token_files: int = None,
    ):
        super().__init__()

        self.token_files = []
        self.n_cond_frames = num_cond_frames
        self.H = H
        self.W = W

        num_token_sequences = []

        for tf in Path(token_root_path).rglob("*.npy"):
            token_tensor = np.load(tf, mmap_mode="r")
            if len(token_tensor.shape) != 3:
                continue
            F, H, W = token_tensor.shape
            if (H != self.H) or (W != self.W) or (F <= num_cond_frames):
                continue

            self.token_files.append(tf)
            num_token_sequences.append(F - num_cond_frames)

            if max_token_files is not None and len(self.token_files) >= max_token_files:
                break

        self.cumulative_lengths = np.cumsum(
            num_token_sequences
        )  # used to map index to token_file

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _map_global_index_to_tensor_index(self, global_idx: int):
        for tnsr_idx, length in enumerate(self.cumulative_lengths):
            if global_idx < length:
                if tnsr_idx == 0:
                    local_idx = global_idx
                else:
                    local_idx = global_idx - self.cumulative_lengths[tnsr_idx - 1]
                return tnsr_idx, local_idx
        raise IndexError("Global index out of range")

    def __getitem__(self, index):
        tnsr_idx, start_frame_idx = self._map_global_index_to_tensor_index(index)
        token_tensor = np.load(self.token_files[tnsr_idx], mmap_mode="r")
        flat_frame_tokens = token_tensor[
            start_frame_idx : start_frame_idx + self.n_cond_frames + 1
        ].reshape(-1)

        context = flat_frame_tokens[:-1]
        target = flat_frame_tokens[1:]

        # convert to torch
        context_tokens = torch.tensor(context, dtype=torch.long)
        target_tokens = torch.tensor(target, dtype=torch.long)

        return context_tokens, target_tokens
