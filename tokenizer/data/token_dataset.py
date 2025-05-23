from typing import Union
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


def increment_slice(s: slice) -> slice:
    return slice(s.start + 1, s.stop + 1, s.step)


class ImageTokenDataset(Dataset):

    def __init__(
        self, token_root_path: Union[str, Path], n_cond_frames: int, H: int, W: int
    ):
        super().__init__()

        self.token_files = []
        self.n_cond_frames = n_cond_frames
        self.H = H
        self.W = W

        num_token_sequences = []

        for tf in Path(token_root_path).rglob("*.npy"):
            token_tensor = np.load(tf, mmap_mode="r")
            if len(token_tensor.shape) != 3:
                continue
            F, H, W = token_tensor.shape
            if (H != self.H) or (W != self.W) or (F <= n_cond_frames):
                continue

            self.token_files.append(tf)
            # calculate the number of token sequences that can be sampled from the tensor
            # there are F-C frames to be predicted
            # for each predicted frame there can H*W token sequences be sampled
            num_token_sequences.append((F - n_cond_frames) * H * W)

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

    def _calc_sequence_index_slice(self, local_idx: int) -> slice:
        n_predictable_sequences = self.H * self.W
        frame_idx = local_idx // n_predictable_sequences
        start_idx = frame_idx * n_predictable_sequences
        end_idx = (
            start_idx
            + n_predictable_sequences * self.n_cond_frames
            + local_idx % n_predictable_sequences
        )
        return slice(start_idx, end_idx)

    def __getitem__(self, index):
        tnsr_idx, local_seq_idx = self._map_global_index_to_tensor_index(index)
        token_sequence_slice = self._calc_sequence_index_slice(local_seq_idx)

        flat_token_tensor = np.load(self.token_files[tnsr_idx], mmap_mode="r").reshape(
            -1
        )
        input_sequence = flat_token_tensor[token_sequence_slice]
        target_sequence = flat_token_tensor[increment_slice(token_sequence_slice)]

        return (
            torch.tensor(input_sequence),
            torch.tensor(target_sequence),
        )
