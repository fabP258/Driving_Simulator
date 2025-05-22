from typing import Union
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


class ImageTokenDataset(Dataset):

    def __init__(self, token_root_path: Union[str, Path]):
        super().__init__()
        self.token_files = list(Path(token_root_path).rglob("*.npy"))

        num_token_sequences = []

        for tf in self.token_files:
            token_tensor = np.load(tf, mmap_mode="r")
            # calculate the number of token sequences that can be sampled from the tensor
            # reduced by 1 to account for the shifted target sequence
            num_token_sequences.append(
                token_tensor.size - (token_tensor.shape[-2] * token_tensor.shape[-1])
            )

        self.cumulative_lengths = np.cumsum(
            num_token_sequences
        )  # used to map index to token_file

        # TODO: derive these from loaded tensors
        self.H = 18
        self.W = 32
        self.sequence_length = self.H * self.W

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

    def _calc_position_indices(
        self, start_index: int, grid: bool = False
    ) -> torch.Tensor:
        """
        Calculates 1D position indices indicating the location of the image patch.
        If `grid==True` the position are rearranged on a 2D grid in row-major order.
        """

        position_indices = torch.tensor(
            [
                (i + start_index) % self.sequence_length
                for i in range(self.sequence_length)
            ],
        )
        if not grid:
            return position_indices
        return position_indices.view(self.H, self.W)

    def __getitem__(self, index):
        tnsr_idx, seq_start_idx = self._map_global_index_to_tensor_index(index)
        token_tensor = np.load(self.token_files[tnsr_idx], mmap_mode="r").reshape(-1)
        input_sequence = token_tensor[
            seq_start_idx : seq_start_idx + self.sequence_length
        ]
        seq_start_idx += 1
        target_sequence = token_tensor[
            seq_start_idx : seq_start_idx + self.sequence_length
        ]
        position_indices = self._calc_position_indices(index)

        return (
            torch.tensor(input_sequence),
            torch.tensor(target_sequence),
            position_indices,
        )
