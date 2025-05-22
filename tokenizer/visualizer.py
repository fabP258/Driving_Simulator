from typing import List
import numpy as np
import torch
from torchvision import transforms
import cv2

from tokenizer.models.vqgan import VQModel
from tokenizer.data.dataset import denormalize_image


class Visualizer:

    def __init__(self, checkpoint_path: str, batch_size: int = 4):
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.model = VQModel.load_from_checkpoint(
            checkpoint_path, sane_index_shape=True
        )
        self.model.eval()
        self.model.to("cuda")

    def encode_and_decode_video(self, video_path: str, output_path: str, fps: int):
        tokens = self.encode_video_as_tokens(video_path)
        reconstructed_frames = self.decode_tokens(tokens)
        self.write_video(reconstructed_frames, output_path, fps)

    def encode_video_as_tokens(self, video_path: str) -> torch.Tensor:
        frames = Visualizer.load_video_frames(video_path)
        tokens = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                frame_batch = torch.stack(frames[i : i + self.batch_size], dim=0).to(
                    self.model.device
                )
                quantized_latents, _, info = self.model.encode(
                    frame_batch.to(self.model.device)
                )
                tokens.append(info[2].cpu())
                del quantized_latents
        return torch.cat(tokens)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        frames = []
        with torch.no_grad():
            for i in range(0, tokens.shape[0], self.batch_size):
                token_batch = tokens[i : i + self.batch_size, :]
                reconstructed_frame_batch = self.model.decode_tokens(
                    token_batch.to(self.model.device)
                )
                frames.append(denormalize_image(reconstructed_frame_batch.cpu()))
                del reconstructed_frame_batch
        return torch.cat(frames, dim=0)

    @staticmethod
    def write_video(frame_tensor: torch.Tensor, video_path: str, fps: int):
        frame_tensor = frame_tensor.detach().numpy()
        frame_tensor = np.transpose(frame_tensor, (0, 2, 3, 1))
        frame_tensor = np.clip(frame_tensor, 0, 1)
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

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((288, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if as_tensor:
                img = preprocess(img)
            frames.append(img)
        return frames
