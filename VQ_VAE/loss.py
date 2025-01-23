import torch
import torch.nn.functional as F
from torch import nn


from discriminator import Discriminator


class GanLoss(nn.Module):

    def __init__(self, num_disc_layers: int, num_disc_hiddens: int):
        super().__init__()

        self.discriminator = Discriminator(
            in_channels=3, num_layers=num_disc_layers, num_hiddens=num_disc_hiddens
        )

    def forward(self, x, x_recon, opt_id: int):

        if opt_id == 0:
            # generator update
            logits_fake = self.discriminator(x_recon)

            # hinge loss
            generator_loss = -torch.mean(logits_fake)

            return generator_loss

        if opt_id == 1:
            # discriminator update
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())  # do not optimize G

            # hinge loss
            loss_real = torch.mean(F.relu(1.0 - logits_real))
            loss_fake = torch.mean(F.relu(1.0 + logits_fake))

            return 0.5 * (loss_real + loss_fake)
