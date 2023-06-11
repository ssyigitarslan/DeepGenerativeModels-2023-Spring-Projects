import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
# VAE Transformer Modeli
class VAETransformer(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, num_layers):
        super(VAETransformer, self).__init__()
        self.encoder_scalar = nn.Linear(input_size, latent_size)
        self.encoder_pairwise = nn.Linear(input_size, latent_size)
        self.transformer_scalar = nn.TransformerEncoder(nn.TransformerEncoderLayer(latent_size, num_heads, dim_feedforward=latent_size), num_layers)
        self.transformer_pairwise = nn.TransformerEncoder(nn.TransformerEncoderLayer(latent_size, num_heads, dim_feedforward=latent_size), num_layers)
        self.decoder_scalar = nn.Linear(latent_size, input_size)
        self.decoder_pairwise = nn.Linear(latent_size, input_size)

    def forward(self, scalar_input, pairwise_input):
        # Encoder
        scalar_latent = self.encoder_scalar(scalar_input)
        pairwise_latent = self.encoder_pairwise(pairwise_input)
        
        # Transformer
        scalar_transformed = self.transformer_scalar(scalar_latent)
        pairwise_transformed = self.transformer_pairwise(pairwise_latent)
        
        # Decoder
        reconstructed_scalar = self.decoder_scalar(scalar_transformed)
        reconstructed_pairwise = self.decoder_pairwise(pairwise_transformed)
        
        return reconstructed_scalar, reconstructed_pairwise, scalar_latent, pairwise_latent