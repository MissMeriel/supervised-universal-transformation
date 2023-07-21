import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Model(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 16, 3, 2),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 2),
            # nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 2),
            # nn.BatchNorm2d(256),
            nn.ELU(),
        )
        encoder_output_shape = self.encoder(torch.ones(1, 3, self.input_shape[0], self.input_shape[1])).shape[1:]
        encoder_output_size = np.product(encoder_output_shape)
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_var = nn.Linear(encoder_output_size, latent_dim)
        print(f"\n{encoder_output_shape=}\n{encoder_output_size=}")

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, encoder_output_size),
            Reshape((-1, *encoder_output_shape)),
            nn.ConvTranspose2d(256, 128, 3, 2, padding=(0,0)),#, output_padding=(1,0)), #0, padding=(1)),
            # nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 0), #, padding=(0, 1)),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 4, 1, padding=(1, 0)),  # , padding=(0, 1)),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 0), #, padding=(1, 0)),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            # nn.ConvTranspose2d(32, 32, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, padding=(1,0)),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, 3, 2),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.ConvTranspose2d(8, 8, 3, 1, padding=(1,0)),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.ConvTranspose2d(8, 3, 3, 2, padding=(0,1)),
            # nn.BatchNorm2d(3),
            # nn.ELU(),
            # nn.ConvTranspose2d(3, 3, 1, 1, padding=(0,1)),
            # nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        print("decoder shape =", self.decoder(torch.ones(1, self.latent_dim)).shape)
        print("decoder size =", np.product(encoder_output_shape))

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x = x.flatten(1)
        result = self.encoder(x).flatten(1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        y = y[...,:-1]
        # y = y.view(-1, 1, 200, 200)
        y = y.view(-1, 3, self.input_shape[0], self.input_shape[1])
        return y

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(self.fc_var.weight.device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


def loss_fn(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    # recons_loss = F.binary_cross_entropy(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss
