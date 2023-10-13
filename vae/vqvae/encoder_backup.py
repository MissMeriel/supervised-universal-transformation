
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vqvae.models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, transf=None, verbose=False):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.verbose = verbose
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )
        if transf == "resinc":
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel + 1,
                          stride=stride + 1, padding=(2, 0)),
                nn.ReLU(),
                nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                          stride=stride + 1, padding=(2, 0)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1,
                          stride=stride - 1, padding=0),
                ResidualStack(
                    h_dim, h_dim, res_h_dim, n_res_layers)
            )
            # elif transf == "resdec":
            #     self.conv_stack = nn.Sequential(
            #         nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
            #                 stride=stride-1, padding=(2, 2)),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
            #                 stride=stride-1, padding=(2, 2)),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim, h_dim, kernel_size=kernel,
            #                 stride=stride, padding=0),
            #         ResidualStack(
            #             h_dim, h_dim, res_h_dim, n_res_layers)
            #     )
            #     self.conv_stack = nn.Sequential(
            #         nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel-2,
            #                 stride=stride-1),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel-1,
            #                 stride=stride-1, padding=0),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
            #                 stride=stride, padding=0),
            #         ResidualStack(
            #             h_dim, h_dim, res_h_dim, n_res_layers)
            #     ) # best so far  x_hat.shape=torch.Size([1, 3, 100, 184])
            #     self.conv_stack = nn.Sequential(
            #         nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel-2,
            #                 stride=stride-1),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel-2,
            #                 stride=stride-1, padding=0),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim, h_dim, kernel_size=kernel-2,
            #                 stride=stride, padding=0),
            #         ResidualStack(
            #             h_dim, h_dim, res_h_dim, n_res_layers)
            #     ) # best so far   x_hat.shape=torch.Size([1, 3, 104, 188])
            #     self.conv_stack = nn.Sequential(
            #         nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel-2,
            #                 stride=stride-1, padding=(1,1)),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel-2,
            #                 stride=stride-1, padding=0),
            #         nn.ReLU(),
            #         nn.Conv2d(h_dim, h_dim, kernel_size=kernel-2,
            #                 stride=stride, padding=0),
            #         ResidualStack(
            #             h_dim, h_dim, res_h_dim, n_res_layers)
            #     ) # best so far   x_hat.shape=torch.Size([1, 3, 104, 188])
        # elif transf == "mediumdepth" or transf == "mediumfisheye":
        #     self.conv_stack = nn.Sequential(
        #         nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel-2,
        #                 stride=stride-1, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel-2,
        #                 stride=stride, padding=0),
        #         nn.ReLU(),
        #         nn.Conv2d(h_dim, h_dim, kernel_size=kernel-2,
        #                 stride=stride, padding=0),
        #         ResidualStack(
        #             h_dim, h_dim, res_h_dim, n_res_layers)
        #     ) # best so far   x_hat.shape=torch.Size([1, 3, 104, 188])

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
