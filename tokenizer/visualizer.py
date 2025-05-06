from trainer import VqVaeTrainer
from typing import List
import numpy as np
import torch
from torchvision import transforms
import cv2


class Visualizer:

    def __init__(self, checkpoint_path: str, batch_size: int = 4):
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.trainer = self.load_trainer(checkpoint_path)

    def encode_and_decode_video(self, video_path: str, output_path: str, fps: int):
        tokens = self.encode_video_as_tokens(video_path)
        reconstructed_frames = self.decode_tokens(tokens)
        self.write_video(reconstructed_frames, output_path, fps)

    def _load_trainer(self, checkpoint_path: str):
        return VqVaeTrainer.from_checkpoint(checkpoint_path, log_path="./")

    def encode_video_as_tokens(self, video_path: str) -> torch.Tensor:
        frames = Visualizer.load_video_frames(video_path)
        tokens = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                frame_batch = torch.stack(frames[i : i + self.batch_size], dim=0)
                quantized_latents, indices = self.trainer.encode(
                    frame_batch.to(self.trainer.device)
                )
                tokens.append(indices.cpu())
                del quantized_latents, indices
        return torch.cat(tokens, dim=0)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        frames = []
        with torch.no_grad():
            for i in range(0, tokens.shape[0], self.batch_size):
                token_batch = tokens[i : i + self.batch_size, :]
                reconstructed_frame_batch = self.trainer.decode_from_indices(
                    token_batch.to(self.trainer.device)
                )
                frames.append(reconstructed_frame_batch.cpu())
                del reconstructed_frame_batch
        return torch.cat(frames, dim=0)

    @staticmethod
    def write_video(frame_tensor: torch.Tensor, video_path: str, fps: int):
        frame_tensor = frame_tensor.detach().numpy()
        frame_tensor = np.transpose(frame_tensor, (0, 2, 3, 1))
        size = (frame_tensor.shape[2], frame_tensor.shape[1])
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(frame_tensor.shape[0]):
            frame = (frame_tensor[i] * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

    @staticmethod
    def load_video_frames(
        video_path: str, as_tensor: bool = True
    ) -> List[torch.Tensor | np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 288))
            if as_tensor:
                transform = transforms.ToTensor()
                img = transform(img)
            frames.append(img)
        return frames
