import torch
from torch import nn
from torch.nn import functional as F

from scipy.cluster.vq import kmeans2


class ExponentialMovingAverage(nn.Module):

    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.register_buffer("value", torch.zeros(*shape))

    @torch.no_grad()
    def update(self, new_value):
        self.value = self.value * self.decay + new_value * (1 - self.decay)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        use_l2_normalization: bool,
        use_ema: bool,
        ema_decay: float,
        ema_eps: float,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_l2_normalization = use_l2_normalization
        self.use_ema = use_ema
        self.ema_eps = ema_eps

        # Dictionary embeddings
        limit = 0.5
        embedding_table = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            # Will be updated by EMA, not the optimizer
            self.register_buffer("embedding_table", embedding_table)
        else:
            # Wille be updated by the optimizer and therefore needs to be returned by self.parameters()
            self.register_parameter("embedding_table", embedding_table)

        self.cluster_count_ma = ExponentialMovingAverage(ema_decay, (num_embeddings,))
        self.embedding_sums = ExponentialMovingAverage(ema_decay, embedding_table.shape)

        self.register_buffer("data_initialized", torch.zeros(1))

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C] -> [B x H x W, C]
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # K means initialization
        # https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
        if self.training and self.data_initialized.item() == 0:
            # Note: This only works for flat_x.shape[0] >= self.num_embeddings!
            centroids, _ = kmeans2(
                flat_x.data.cpu().numpy(),
                self.num_embeddings,
                minit="points",
            )
            self.embedding_table.data.copy_(torch.from_numpy(centroids.T))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        embedding_table = self.embedding_table
        if self.use_l2_normalization:
            flat_x = F.normalize(flat_x, p=2, dim=1)
            embedding_table = F.normalize(embedding_table, p=2, dim=0)
        distances = (
            (flat_x**2).sum(1, keepdim=True)
            - 2 * flat_x @ embedding_table
            + (embedding_table**2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)

        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]),
            self.embedding_table.transpose(0, 1),
        ).permute(0, 3, 1, 2)

        dictionary_loss = None
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()

        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()

        # Straight-through gradient.
        quantized_x = x + (quantized_x - x).detach()

        # Entropy of codebook selection distribution
        codebook_usage_counts = torch.bincount(encoding_indices.flatten())
        codebook_probabilities = codebook_usage_counts / codebook_usage_counts.sum()
        entropy = torch.sum(
            codebook_probabilities * torch.log(codebook_probabilities + 1e-10)
        )

        if self.use_ema and self.training:
            with torch.no_grad():
                # Cluster count moving average update
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                self.cluster_count_ma.update(encoding_one_hots.sum(0))

                # Embedding moving average update
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.embedding_sums.update(embed_sums)

                # apply eps on normalized counts (then denormalize)
                cluster_count_sum = self.cluster_count_ma.value.sum()
                nonzero_cluster_counts = (
                    (self.cluster_count_ma.value + self.ema_eps)
                    / (cluster_count_sum + self.num_embeddings * self.ema_eps)
                    * cluster_count_sum
                )

                self.embedding_table = (
                    self.embedding_sums.value / nonzero_cluster_counts
                )

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            entropy,
            encoding_indices.view(x.shape[0], -1),
        )
