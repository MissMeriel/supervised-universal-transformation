
import torch
import torch.nn as nn
import numpy as np
import sys, os

# sys.path.append(os.getcwd()+"/..")
# sys.path.append(os.getcwd()+"/../..")
sys.path.append("../../vqvae/models")
from vqvae.models.encoder import Encoder
from vqvae.models.quantizer import VectorQuantizer
from vqvae.models.decoder import Decoder

print(os.getcwd())
class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, transf=None, arch_id=None):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, transf, arch_id)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, transf, arch_id)
        
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            # print('decoder input shape:', z_q.shape)
            print('recon data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 3, 270, 480))
    x = torch.tensor(x).float()

    # test vqvae
    vqvae = VQVAE()
    embedding_loss, x_hat, perplexity = vqvae(x)
    print('VQVAE out shape:', x_hat.shape)
