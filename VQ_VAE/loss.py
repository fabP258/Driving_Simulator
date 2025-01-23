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
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.disc_factor = 1.0
        self.discriminator_weight = 1.0

    def calculate_adaptive_weight(self, rec_loss, gen_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, x, x_recon, opt_id: int, recon_loss, last_layer):

        if opt_id == 0:
            # generator update
            logits_fake = self.discriminator(x_recon)

            # hinge loss
            # generator_loss = -torch.mean(logits_fake)

            generator_loss = self.loss_fn(logits_fake, torch.zeros_like(logits_fake))

            d_weight = self.calculate_adaptive_weight(
                recon_loss, generator_loss, last_layer
            )

            return generator_loss * d_weight * self.disc_factor

        if opt_id == 1:
            # discriminator update
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())  # do not optimize G

            # hinge loss
            # loss_real = torch.mean(F.relu(1.0 - logits_real))
            # loss_fake = torch.mean(F.relu(1.0 + logits_fake))

            loss_real = self.loss_fn(logits_real, torch.ones_like(logits_real))
            loss_fake = self.loss_fn(logits_fake, torch.zeros_like(logits_fake))

            return 0.5 * (loss_real + loss_fake) * self.disc_factor
