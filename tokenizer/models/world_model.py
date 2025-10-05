import math
import torch
from typing import Optional
from torchvision.utils import make_grid

from tokenizer.engine.module import TrainableModule
from tokenizer.modules.transformer.model import Transformer

#from cosmos_tokenizer.image_lib import ImageTokenizer


class WorldModel(TrainableModule):

    @TrainableModule.save_hyperparameters
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        n_cond_frames: int,
        H: int,
        W: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        vocab_size: int,  # TODO: maybe give this a better name like image_vocab_size
        bias: bool,
        dropout: float,
        decoder_ckpt: str,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        block_size = n_cond_frames * H * W + (H * W - 1)
        self.warm_up_steps = 200
        self.total_steps = 2000000
        self.transformer = Transformer(
            vocab_size, block_size, n_embd, n_head, bias, n_layer, dropout
        )
        #self.tokenizer = ImageTokenizer(checkpoint_dec=decoder_ckpt, device="cpu")
        self.tokenizer = None

    def _training_step(self, batch, batch_idx):
        logits, loss = self.transformer(*batch)
        # TODO: calculate loss here
        self.log_scalar("train/cross_entropy_loss", loss.item(), step=self.step)
        self.log_scalar(
            "train/learning_rate",
            self.lr_schedulers[0].get_last_lr()[0],
            step=self.step,
        )
        if (batch_idx % 100) == 0:
            self.log_frame_sequences(logits.detach(), batch[0], max_context_frames=None)
        return loss

    @torch.no_grad
    def log_frame_sequences(
        self,
        logits: torch.Tensor,
        input_tokens: torch.Tensor,
        max_context_frames: Optional[int] = None,
    ):
        B = logits.size(0)
        H, W = self.hparams["H"], self.hparams["W"]
        T = self.hparams["n_cond_frames"]
        max_context_frames = (
            T if max_context_frames is None else min(max_context_frames, T)
        )
        num_cond_tokens = T * H * W

        self.tokenizer = self.tokenizer.to("cuda")

        # decode predicted frames
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_tokens = torch.argmax(probs, dim=-1)
        self.logger.add_histogram(
            "train/output_token_distribution",
            pred_tokens.view(-1).cpu(),
            global_step=self.step,
        )
        self.logger.add_histogram(
            "train/input_token_distribution",
            input_tokens.view(-1).cpu(),
            global_step=self.step,
        )
        pred_tokens = pred_tokens[
            :,
            (num_cond_tokens - 1) :,
        ]
        pred_tokens = pred_tokens.reshape(B, H, W)
        pred_imgs = self.tokenizer.decode(pred_tokens).cpu()  # (B, C, H, W)

        # decode conditioning frames
        cond_frame_tokens = input_tokens[:, :num_cond_tokens]
        cond_frame_tokens = cond_frame_tokens.reshape(B, T, H, W)
        cond_frame_tokens = cond_frame_tokens[:, -max_context_frames:]
        cond_imgs = []
        for t in range(max_context_frames):
            img = self.tokenizer.decode(cond_frame_tokens[:, t]).cpu()
            cond_imgs.append(img)
        cond_imgs = torch.stack(cond_imgs, dim=1)  # (B, T, C, H, W)

        # combine cond. frames and pred. frame
        pred_imgs = pred_imgs.unsqueeze(1)  # (B, 1, C, H, W)
        all_imgs = torch.cat([cond_imgs, pred_imgs], dim=1)  # (B, T+1, C, H, W)
        all_imgs = all_imgs.permute(0, 2, 3, 1, 4)  # (B, C, H, T+1, W)
        combined_imgs = all_imgs.reshape(
            B, all_imgs.shape[1], all_imgs.shape[2], -1
        )  # (B, C, H, (T+1)*W)

        grid_img = make_grid(
            combined_imgs, nrow=1, normalize=True, value_range=(-1, 1)
        )  # stack vertically

        self.logger.add_image("train/predicted_frame", grid_img, global_step=self.step)
        self.tokenizer = self.tokenizer.cpu()

    def forward(self, x):
        return self.transformer(x)

    @torch.no_grad
    def generate_frames(
        self,
        idx: torch.Tensor,
        n_frames: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        tokens_per_frame = self.hparams["H"] * self.hparams["W"]
        n_cond_tokens = (
            tokens_per_frame
            * self.hparams["n_cond_frames"]  # TODO: store these explicitly
        )
        # check if the input sequence length is a multiple of the frame size
        assert idx.shape[1] % (self.hparams["H"] * self.hparams["W"]) == 0
        # check if the input sequence length is equal or larger as the number of cond. tokens
        assert idx.shape[1] >= n_cond_tokens  # TODO: force equality?

        for _ in range(n_frames):
            for i in range(tokens_per_frame):
                # context = conditioning frames + generated tokens of next frame
                idx_cond = idx[:, -(n_cond_tokens + i) :]
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, betas=(0.9, 0.95), fused=True
        )

        def lr_lambda(current_step):
            if current_step < self.warm_up_steps:
                return float(current_step) / float(max(1, self.warm_up_steps))
            progress = float(current_step - self.warm_up_steps) / float(
                max(1, self.total_steps - self.warm_up_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]
