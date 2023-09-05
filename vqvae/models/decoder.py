
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


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

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, transf=None, verbose=False):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.verbose = verbose
        # self.inverse_conv_stack = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
        #     ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        #     nn.ConvTranspose2d(h_dim, h_dim // 2,
        #                        kernel_size=kernel, stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
        #                        stride=stride, padding=1)
        # )

        self.convtrans1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        self.convtrans2 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        if transf == "resdec":
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel//4, stride=stride//2, padding=1) # use for 135 X 240 to resdec 67 X 120, output recon data shape: torch.Size([1, 3, 64, 118]) 
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=3, stride=stride, padding=1)
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel*2, stride=stride*2, padding=1) # use for input 67 X 120 to output 135 X 240, returns torch.Size([1, 3, 130, 242])
            #self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=int(kernel*1.5), stride=int(stride*1.5), padding=1) # use for input 67 X 120 to output 108 X 192, outputs torch.Size([1, 3, 97, 181])
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=7, stride=3, padding=0) # BETTER for input 67 X 120 to output 108 X 192, outputs torch.Size([1, 3, 100, 184])
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=8, stride=3, padding=0) # EVEN BETTER for input 67 X 120 to output 108 X 192, outputs torch.Size([1, 3, 101, 185])
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=9, stride=3, padding=0) # EVEN EVEN BETTER for input 67 X 120 to output 108 X 192, outputs torch.Size([1, 3, 102, 186])
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=16, stride=3, padding=0) # Best for input 67 X 120 to output 108 X 192, outputs torch.Size([1, 3, 109, 193])
            self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=8, stride=4, padding=0) # 54 x 96 to 108 x 192, output is torch.Size([1, 3, 108, 196])
        elif transf == "resinc":
            # self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel**3, stride=stride**2, padding=1) # intended resinc 270 x 480, output recon data shape: torch.Size([1, 3, 274, 490])
            self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel//2, stride=stride, padding=1) # 480 X 270 to 108 x 192, output is torch.Size([1, 3, 266, 478])
        else:
            self.convtrans3 = nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel, stride=stride, padding=1) # original one


    def forward(self, x):
        y = self.convtrans1(x)
        y = self.residual_stack(y)
        y = self.convtrans2(y)
        y = nn.ReLU()(y)
        y = self.convtrans3(y)
        return y


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
