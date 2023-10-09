
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vqvae.models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, transf=None, arch_id=None, verbose=False):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.verbose = verbose
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, 
                               kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, 
                               kernel_size=kernel, stride=stride, padding=1)
        )
        
        # original decoder
        # self.convtrans1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        # self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        # self.convtrans2 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel, stride=stride, padding=1)
        # self.transf = transf
        # if transf == "resdec":
        #     # self.convtrans3 = nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=16, stride=3, padding=0)
        #     self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=8, stride=4, padding=0) # 54 x 96 to 108 x 192, output is torch.Size([1, 3, 108, 196])
        # elif transf == "resinc":
        #     self.convtrans1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=32, stride=1, padding=(2, 0))
        #     self.convtrans2 = nn.ConvTranspose2d(h_dim, h_dim//2, kernel_size=32, stride=1, padding=(8, 0))
        #     self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=16, stride=1, padding=(8, 2)) # 480 X 270 to 108 x 192, output is torch.Size([1, 3, 266, 478])


    def forward(self, x):
        return self.inverse_conv_stack(x)
        # y = self.convtrans1(x)
        # y = self.residual_stack(y)
        # y = self.convtrans2(y)
        # y = nn.ReLU()(y)
        # y = self.convtrans3(y)
        # return y


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
