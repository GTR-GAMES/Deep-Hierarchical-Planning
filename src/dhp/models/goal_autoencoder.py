import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GoalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.matrix_width = 8  # Fixed per paper
        self.num_codes = 8     # Categories per row

        # Encoder (direct projection)
        self.encoder = nn.Linear(input_dim, self.matrix_width * self.num_codes)

        # Decoder (direct projection)
        self.decoder = nn.Linear(self.matrix_width * self.num_codes, input_dim)

    def forward(self, x):
        # Get logits [batch, 8*8]
        logits = self.encoder(x)

        # Reshape to [batch, 8, 8]
        logits = logits.view(-1, self.matrix_width, self.num_codes)

        # Sample z [batch, 8, 1]
        z = torch.multinomial(
            F.softmax(logits, dim=-1).view(-1, self.num_codes), 1)
        z = z.view(-1, self.matrix_width, 1)

        # Straight-through estimation
        z_hard = torch.zeros_like(logits).scatter_(-1, z, 1.0)
        z = z_hard - F.softmax(logits, dim=-1).detach() + \
            F.softmax(logits, dim=-1)

        # Flatten to [batch, 64]
        z_flat = z.view(-1, self.matrix_width * self.num_codes)

        # Decode
        reconstructed = self.decoder(z_flat)
        return logits, z_flat, reconstructed

    def loss(self, decoded, original, logits, beta=1.0):
        # Reconstruction loss
        # print(f"{decoded.shape=}, {original.shape=}")
        recon_loss = F.mse_loss(decoded, original)

        # KL divergence per row
        log_probs = F.log_softmax(logits, dim=-1)
        uniform_prior = torch.full_like(log_probs, np.log(1/self.num_codes))
        kl_loss = F.kl_div(log_probs, uniform_prior, reduction='none')
        # Sum over categories, mean over batches/rows
        kl_loss = kl_loss.sum(dim=-1).mean()

        return recon_loss + beta * kl_loss
