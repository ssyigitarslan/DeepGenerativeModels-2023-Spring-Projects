import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_v1(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE_v1, self).__init__()
        #print("asd1")
        # Encoder
        self.conv = nn.Conv1d(input_size, latent_size, kernel_size=1, dilation=2)  # Dilated convolution
        self.fc_mu = nn.Linear(latent_size, latent_size)  # To calculate mean
        self.fc_logvar = nn.Linear(latent_size, latent_size)  # To calculate log variance
        self.transformer = nn.Transformer(latent_size, nhead=8)  # Transformer layer
        #print("asd2")
        # Decoder
        self.fc_dec = nn.Linear(latent_size, latent_size)
        self.conv_transpose = nn.ConvTranspose1d(latent_size, input_size, kernel_size=1, dilation=2)  # Transposed convolution

    def encode(self, x):
        x = F.relu(self.conv(x))
        #print("asd3")
        x = x.view(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #print("asd4")
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc_dec(z))
        z = z.view(z.size(0), z.size(1), 1)  # Reshape the tensor to a suitable shape for transposed convolution
        #print("asd5")
        return torch.sigmoid(self.conv_transpose(z))

    def forward(self, scalar_input, pairwise_input):
        # Encoder
        scalar_latent, scalar_log_var = self.encode(scalar_input)
        pairwise_latent, pairwise_log_var = self.encode(pairwise_input)
        #print("asd6")

        # Reparameterization
        scalar_z = self.reparameterize(scalar_latent, scalar_log_var)
        pairwise_z = self.reparameterize(pairwise_latent, pairwise_log_var)

        # Transformer
        scalar_z = self.transformer(scalar_z.unsqueeze(0), tgt=scalar_z.unsqueeze(0)).squeeze(0)
        pairwise_z = self.transformer(pairwise_z.unsqueeze(0), tgt=pairwise_z.unsqueeze(0)).squeeze(0)
        #print("asd7")

        # Decoder
        reconstructed_scalar = self.decode(scalar_z)
        reconstructed_pairwise = self.decode(pairwise_z)

        return reconstructed_scalar, reconstructed_pairwise, scalar_latent, pairwise_latent
