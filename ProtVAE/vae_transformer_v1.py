import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, GRU

class VAETransformer_v1(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, num_layers):
        super(VAETransformer_v1, self).__init__()

        self.encoder_scalar = nn.Linear(input_size, latent_size)
        self.encoder_pairwise = nn.Linear(input_size, latent_size)

        transformer_layer_scalar = TransformerEncoderLayer(d_model=latent_size, nhead=num_heads)
        transformer_layer_pairwise = TransformerEncoderLayer(d_model=latent_size, nhead=num_heads)

        self.transformer_scalar = TransformerEncoder(transformer_layer_scalar, num_layers=num_layers)
        self.transformer_pairwise = TransformerEncoder(transformer_layer_pairwise, num_layers=num_layers)

        self.gru_scalar = GRU(latent_size, latent_size, batch_first=True)
        self.gru_pairwise = GRU(latent_size, latent_size, batch_first=True)

        self.decoder_scalar = nn.Linear(latent_size, input_size)
        self.decoder_pairwise = nn.Linear(latent_size, input_size)

    def forward(self, scalar_input, pairwise_input):
        # Encoder
        scalar_latent = self.encoder_scalar(scalar_input)
        pairwise_latent = self.encoder_pairwise(pairwise_input)
        
        # Transformer
        scalar_transformed = self.transformer_scalar(scalar_latent.unsqueeze(0))
        pairwise_transformed = self.transformer_pairwise(pairwise_latent.unsqueeze(0))

        # Autoregression with GRU
        _, scalar_hidden = self.gru_scalar(scalar_transformed)
        _, pairwise_hidden = self.gru_pairwise(pairwise_transformed)

        # Decoder
        reconstructed_scalar = self.decoder_scalar(scalar_hidden.squeeze(0))
        reconstructed_pairwise = self.decoder_pairwise(pairwise_hidden.squeeze(0))
        
        return reconstructed_scalar, reconstructed_pairwise, scalar_latent, pairwise_latent