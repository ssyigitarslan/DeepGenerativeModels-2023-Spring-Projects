"""

GENERATOR.PY consists of the following classe definitions:

1.       Generator
1.1.     MappingNetwork
1.2.     SynthesisNetwork
1.2.1.   EncoderBlock 
1.2.1.1. EncoderLayer (with Attention)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EqualizedLinear, ToRGBLayer, UpSample

"""
1. Generator

Generator is basically a bundle of Synthesis Network and Mapping Network, taking the style condition given by Mapping Network and feeding it to number of encoder blocks.
"""

class Generator(torch.nn.Module):
    def __init__(self, 
                 h_dim_w,                    # Intermediate latent (W) dimensionality. 
                 w_depth,                    # Number of mapping layers 
                 im_resolution,              # Output resolution. --> Default is 32 for CIFAR10
                 attn_depth,                 # Depth of the attention, --> Setting it 32 is a sweetspot according to fig.3
                 num_enc_blocks,             # List of number of blocks in the systhesis network --> Default=[1,3,3]; simplified=[1,2,2]  
                 enc_dims,                   # List of hidden dimension of transformer encoder_layer for resolution, default=[1024, 512, 512] --> simplified[256,64,16] 
                 ):
      
        super().__init__()

        # Synthesis network, as in StyleGAN2, generates an image given the style conditions W
        self.synthesis = SynthesisNetwork(
            h_dim_w=h_dim_w, im_resolution=im_resolution, 
            attn_depth=attn_depth, num_enc_blocks=num_enc_blocks,
            enc_dims=enc_dims
            )
        
        self.num_blocks_to_feed_W = self.synthesis.num_blocks_to_feed_w
        self.mapping = MappingNetwork(h_dim_w=h_dim_w, w_depth=w_depth)       
        
    def forward(self, z):
        # Given a noise vector z:
        # First, get the mapping network's output W, which will be used for style vectors
        W = self.mapping(z)

        # Expand the W such that every block of Generator will receieve a copy
        W = W.unsqueeze(1).repeat([1, self.num_blocks_to_feed_W, 1]) 
        
        # Feed the Ws to the generative network to synthesize a batch of images
        return self.synthesis(W)


"""
1.1. MappingNetwork

**Disclaimer:** This class is moslty based on [LABML's](https://nn.labml.ai/gan/stylegan/index.html) implementation. See [their description](https://nn.labml.ai/gan/stylegan/index.html#equalized_weights) for details. 

Essentially, Mapping Network takes a noise input, and passes the noise vector through an MLP, to obtain W vector which will be used to obtain Style vectors in the encoder.
"""

class MappingNetwork(torch.nn.Module):
    def __init__(self, h_dim_w, w_depth):
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(w_depth):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(h_dim_w, h_dim_w))
            # Leaky Relu
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)


"""
1.2. SynthesisNetwork

Similar to StyleGAN2 implementation, this class is used as a bundle of encoders that generate images sequentially. 

**Disclaimer:** This network is based on original [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py) implementation. Styleformer claims that they put their extensions on top of  StyleGAN2-ADA implementation, so we also used StyleGAN2-ADA implementation together with LABML's simplified StyleGAN2 and extended their classes according to Styleformer's paper.
"""

class SynthesisNetwork(nn.Module):
    # num_enc_blocks is the "Layers" in Table 5
    # enc_dims is the "Hidden Size" in Table 5
    # h_dim_w is the hidden size of mapping network output w
    # Originally num_enc_blocks should be [1, 3, 3] and
    # enc_dims = [1024, 512, 512] (see Table 5); however, we cannot fit
    # this model to the Colab's RAM. Therefore, we have adjusted the defaults.
    def __init__(self, h_dim_w, im_resolution=32, 
                 attn_depth=32, num_enc_blocks=[1,2,2], enc_dims=[256,64,16]):
              
        super().__init__()    
        self.h_dim_w = h_dim_w
        self.im_resolution = im_resolution
        self.im_resolution_log2 = int(np.log2(im_resolution))
        self.block_count_list = num_enc_blocks
        
        # Block resolutions start from 8x8 and goes up to 32x32 for CIFAR10
        # log of image resolution is taken for computing the resolutions for each stage
        # self.encoder_resolutions = [2 ** i for i in range(3, self.im_resolution_log2 + 1)]
        self.encoder_resolutions = 2 ** np.round(np.linspace(3, self.im_resolution_log2, num=len(enc_dims), dtype=None))
        self.encoder_resolutions[-1] = self.im_resolution
        self.encoder_resolutions = self.encoder_resolutions.astype(int)

        self.num_blocks_to_feed_w = 0
        for i, resolution in enumerate(self.encoder_resolutions):
            h_size  = enc_dims[i] 
            next_dim = None
            if resolution!=self.im_resolution:
                next_dim = enc_dims[i+1]       # Note that i+1 stops at enc_dims[-1], not causing index range problems

            block_count = self.block_count_list[i]

            pos_flag = 2                      # Flag that will indicate the location of the encoder (not the exact index, just an indication of head or tail)
            for j in range(block_count):

                if j == 0:                    # If this is the encoder block at the beginning
                  pos_flag = 0
                if j == block_count - 1:      # If this is the encoder block at the end
                  pos_flag = -1
                if block_count == 1:          # If there is only one encoder block, that block is the first and last block of that encoder resolution
                  pos_flag = -2
                
                block = EncoderBlock(
                                   h_size = h_size , 
                                   h_dim_w = h_dim_w, 
                                   out_dim = next_dim, 
                                   attn_depth = attn_depth, 
                                   im_resolution = im_resolution, 
                                   resolution = resolution, 
                                   pos_flag = pos_flag 
                                    )
                
                # Counting the number of block that will receive mapping network's W
                # Add one for each encoder
                self.num_blocks_to_feed_w += 1 
                # For the last encoder block, toRGB layer will also take one
                if (pos_flag == -1 or pos_flag == -2): # TODO: pos_flag < 0 ?
                    self.num_blocks_to_feed_w += 1

                # Set the encoder blocks as an attribute of Synthesis network
                # so that we can save and load the model according to dictionary.
                setattr(self, f'b{resolution}_{j}', block)

    # Input of Synthesis module is the output of Mapping Network, W
    # Note that W is expanded to match with number of blocks that will
    # receive the mapping vector. All blocks get the same W, pass this 
    # vector to different affine networks to obtain different style vectors.
    def forward(self, W):

        W = W.to(torch.float32)
        k = 0 # Iterator for W
        w_for_each_block = []
        for i, resolution in enumerate(self.encoder_resolutions):
            block_count = self.block_count_list[i]
            w_for_inside = []

            for j in range(block_count):
                block = getattr(self, f'b{resolution}_{j}')
                # Two is for toRGB layers and encoder blocks
                # Note that toRGB layers also take mapping network's output
                w_for_inside.append(W[:,k:k+2,:])       
                k += 1 
                
            w_for_each_block.append(w_for_inside)                                            
  
        i = 0
        x, img = None, None 
        for i in range(len(w_for_each_block)):

          resolution = self.encoder_resolutions[i]
          block_count = self.block_count_list[i]
          w_for_enc = w_for_each_block[i]

          for block_idx in range(block_count):
            block = getattr(self, f'b{resolution}_{block_idx}')
            x, img = block(x, img, w_for_enc[block_idx])
                      
        return img


"""
1.2.1. EncoderBlock

This class is the implementation of entire fig.2(b), plus upsampling and tRGB blocks of fig.2(a)
"""


class EncoderBlock(nn.Module):
    def __init__(self, h_size, h_dim_w, out_dim, attn_depth, im_resolution, resolution, pos_flag):
        super().__init__()

        self.up_sample = UpSample()

        self.h_size = h_size                                # Hidden size of the encoder (see Table 5)
        self.h_dim_w = h_dim_w                              # Hidden size of mapping network's output

        self.out_dim = out_dim                              # Dimension of the output vector
        self.attn_depth = attn_depth                        # Depth of the attention (32 is recommended according to fig.3 results)

        self.im_resolution = im_resolution                  # Original image resolution, 32 for CIFAR10
        self.enc_resolution = resolution                    # Encoder's resolution at the current block, e.g. 8, 16, or 32 for CIFAR10

        self.pos_flag = pos_flag                            # This flag tells whether this is the first (0) block of entire encoder
                                                            # (entire encoder is the blue box of fig.2(a))
                                                            # or it is the encoder block at the end (-1)   
                                                            # Note that Encoder Block class is not the Encoder (blue box of fig.2(a))
                                                            # Rather it is the block at fig.2(b) plus toRGB and upsampling layers
        # If we are not at the last encoder block of an encoder, 
        # output dimension will be the same as current model's dimension
        if not (pos_flag == -1 or pos_flag == -2) or out_dim is None:
            self.out_dim = h_size

        # If this is the first encoder block of the entire Encoder (Encoder is the blue box of fig.2(a) 
        # where it has L, M, and N... encoders inside) add the positional embeddings.
        # Plus, if this is the very first encoder block, i.e. first encoder block of the first encoder
        # provide a constant embedding as an input, just like in StyleGAN2.
        self.positional_embed = None
        self.input_constant = None

        if (pos_flag == 0 or pos_flag == -2):
          self.positional_embed = torch.nn.Parameter(torch.zeros(1, resolution * resolution, self.h_size))
          if self.enc_resolution == 8:
            self.input_constant = torch.nn.Parameter(torch.randn([8 * 8, self.h_size]))
        
        self.enc = EncoderLayer(h_size=self.h_size, 
                                h_dim_w=self.h_dim_w, 
                                out_dim=self.out_dim, 
                                flattened_im_dims=resolution * resolution, 
                                attn_depth=self.attn_depth, 
                                )        
        if (pos_flag == -1 or pos_flag == -2):    # TODO: pos_flag < 0 ? 
            # Note: 3 is the number of channels, i.e. {R, G, B}
            # This will be the green box in fig.2(a) for every last encoder block of an encoder
            self.t_rgb = ToRGBLayer(self.out_dim, 3, h_dim_w=h_dim_w)

    def forward(self, x, img, ws):
                
        # If this is the very first encoder block of resolution 8x8xC (first block in fig.2 (a))
        # Then we start with constant, otherwise the provided x will be used.
        if self.input_constant is not None:
            x = self.input_constant.reshape(1, self.input_constant.shape[0], self.input_constant.shape[1]).repeat([ws.shape[0], 1, 1])
        
        # Add the positional embedding to the first endocer block 
        # (there will be other L-1 blocks, where L is the repeated encoder blocks of fig.2 (a))
        # (see "xL", "xM" etc. next to the encoders in fig.2 (a))
        if self.positional_embed is not None:
            x = x + self.positional_embed
        
        x = self.enc(x, ws[:,0,:])
        
        # In the last encoder layer, the output will be converted to RGB with ToRGB layers
        if (self.pos_flag == -1 or self.pos_flag == -2):  # TODO: pos_flag < 0 ?
            y = self.t_rgb(x.transpose(1,2).reshape(ws.shape[0], self.out_dim, self.enc_resolution, self.enc_resolution),ws[:,1,:])

            if img is None:
              # This is the very first encoder with L blocks
              # At the last layer, we don't have any incoming images
              # so we will just use the output of tRGB. 
              # See the left flow at the fig.2 (a)
              img = y 
            else:
              # Here we are at the second or later encoder's last block
              # Not only we convert the encoder output to RGB image,
              # We also upsample the previous RGB image and add them up.
              img = self.up_sample(img)
              img = img.add_(y) 

            # We then Upsample the RGB image
            # This flow is slightly different than the reference StyleGAN2 implementation
            # Notice that each encoder block's output (actually the last repeated encoder block's)
            # will be unflattened and fed to tRGB in fig.2(a) then they will be upsampled.
            # The unflattened input also will be upsampled for the next encoder block
            # until we reach the original image resolution.
            if self.enc_resolution != self.im_resolution:
                x = self.up_sample(x.transpose(1,2).reshape(ws.shape[0], self.out_dim, self.enc_resolution, self.enc_resolution))
                x = x.reshape(ws.shape[0], self.out_dim, (self.enc_resolution**2) * 2 **2).transpose(1,2)
               
        return x, img


"""
1.2.1.1. EncoderLayer (with Attention)

On top of the original attention mechanism, we have implemented the Styleformer's attention mechanism which includes modulating and demodulating Query, Key, and Value. 

The flow of Multi-Head Attention module is referenced from [this implementation](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)
"""

class EncoderLayer(nn.Module):
    def __init__(self, h_size, h_dim_w, out_dim, flattened_im_dims, attn_depth):
        super().__init__()
        self.h_size  = h_size 
        self.num_heads = int(h_size / attn_depth)
        self.attn_depth = attn_depth 
        # TODO: Maybe assert that depth = h_size // num_heads?

        # Making sure the number of heads 
        # TODO: can we assert h_size  < depth and remove below?
        if(self.num_heads < 1): 
          self.num_heads = 1

        # According the Styleformer paper, less than 32 depth drops the performance
        # in fact, depth = 32 is a sweet-spot for Styleformer, see fig.3
        if attn_depth < 32:
          print(">> WARNING: Attention depth is less than 32.")
      
        self.flattened_im_dims =  flattened_im_dims
        
        # Pass the Mapping Network's output W through a linear layer
        # that gives us a Style vector. Even though we use same W throughout
        # the entire network, different style vectors are computed with different
        # linear layers (or as paper states: different *affine transformations* as
        # the linear layer's weight matrix describes an affine transformation and
        # we won't be passing the output through a non-linear layer like ReLU).
        #
        # See the gray boxes in fig.2 (b)
        # Style Input is the Style vector which will be modulated with input
        # Similarly, Style Value is the Style vector to be modulated with Value of attention
        # h_dim_w is the hidden size of mapping network output w
        # h_size is the hidden size of the encoder (see Table 5, for naming)
        self.to_style_input = EqualizedLinear(h_dim_w, h_size)
        self.to_style_value = EqualizedLinear(h_dim_w, h_size )
        
        # Here, instead of defining linear layers for Q, K, V as in the reference link, 
        # we define torch parameters as we will compute modulation statistics out of these weights.
        self.W_q = torch.nn.Parameter(torch.FloatTensor( h_size, h_size ))
        self.W_k = torch.nn.Parameter(torch.FloatTensor( h_size, h_size ))        
        self.W_v = torch.nn.Parameter(torch.FloatTensor( h_size, h_size ))
        self.W_attn = torch.nn.Parameter(torch.FloatTensor(out_dim, h_size ))
        self.W_residual = torch.nn.Parameter(torch.FloatTensor(out_dim, h_size )) 
        
        # TODO: How to initialize bias?
        self.bias = torch.nn.Parameter(torch.zeros([out_dim]))

        # Initialize weights
        # Actually, authors mention "We initialize all weights in Styleformer
        # encoder using same method used in Pytorch linear layer."
        # TODO: Check how Pytorch initializes linear layer.
        torch.nn.init.xavier_uniform_(self.W_q)
        torch.nn.init.xavier_uniform_(self.W_k)
        torch.nn.init.xavier_uniform_(self.W_v)
        torch.nn.init.xavier_uniform_(self.W_attn)
        torch.nn.init.xavier_uniform_(self.W_residual)
        #torch.nn.init.xavier_uniform_(self.bias)

        
    def forward(self, x, w):
        
        styles_input = self.to_style_input(w)
        styles_value = self.to_style_value(w)
        
        # Use the input to compute attention map
        x = self.encoder_attention(x=x, styles_input=styles_input, styles_value = styles_value, num_heads=self.num_heads)   
        
        # According to fig.2 (b), at the end of the encoder block, we add noise and bias
        # TODO: Should we learn the noise like bias? And what is the range of noise?
        x = x + self.bias 

        noise = torch.randn([x.shape[0], self.flattened_im_dims, 1], device = x.device) * 1e-6
        x = x + noise
        
        # Pass the final output through Leaky ReLU with alpha=0.2 (see Appendix.A)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x

    # This part, encoder_attention, is the majority of the encoder block in fig.2(b), 
    # excluding additional bias and noise.
    def encoder_attention(self, x, styles_input, styles_value, num_heads):
    
      batch_size, flattened_im_len, hidden_dimension = x.shape
      
      # Remember Style vectors had shape [batch_size, hidden_dims_of_encoder]
      # which we obtained after a linear layer with weights [h_dim_w, hidden_enc]
      # 
      # Now we are going to expand one dimension to the style vectors in order to broadcast
      # during modulation multiplication.
      # E.g. for every instance *i* in the batch, styles_val[i] has shape [1, hidden_size]
      # and the weights of Q,K,V has shape [hidden_size, hidden_size]
      # We will obtain [1, hidden_size] modulated vectors for every instance *i* in a batch
      styles_val = styles_value.reshape(batch_size, 1, -1)
      styles_in = styles_input.reshape(batch_size, 1, -1)  
      
      attn_depth = int(hidden_dimension / num_heads)

      # Modulate the input with Style Input, i.e. grey box in fig.2 (b)
      x = x * styles_in
      
      # See fig.2(b), Pre-Layernorm highlighted in a yellow box
      # Note that authors do not give specific details about
      # how the layer normalization is performed.
      x = F.layer_norm(x, x.shape)
      
      # Instead of modulating the Q,K,V, we can modulate their weights
      # as stated both in StyleGAN2 and Styleformer.
      # In order to demodulate Q,K,V, we need the modulated weights of them
      # according to eqn.(1).
      # The weights has shape [hidden_dims, hiddem_dims],
      # The trick is to add a new dimension to make them [1, h_dims, h_dims]
      # And style vectors also have [batch_size, 1, h_dims]
      # In the end, we want [batch_size, h_dims, h_dims] feature map that related
      # every input feature to every other feature.
      Q_mod_w = styles_in * self.W_q.reshape(1, self.W_q.shape[0], self.W_q.shape[1])
      K_mod_w = styles_in * self.W_k.reshape(1, self.W_k.shape[0], self.W_k.shape[1])
      V_mod_w = styles_in * self.W_v.reshape(1, self.W_v.shape[0], self.W_v.shape[1])
      # The variables above gives us the pink boxes of fig.2(b)
      
      # Demodulation weights that will divide the term to be demodulated
      # See eqn. (1) and (2) and their descriptions for background.
      sigma_j_q = torch.sqrt(torch.square(Q_mod_w).sum(dim=2))
      sigma_j_k = torch.sqrt(torch.square(K_mod_w).sum(dim=2))
      sigma_j_v = torch.sqrt(torch.square(V_mod_w).sum(dim=2))

      # Now we have the standard deviation of modulated weights 
      # Which we will use them to divide the modulated input x,
      # and obtain red boxes of fig.2(b)
      Q = torch.matmul(x, self.W_q.t())
      K = torch.matmul(x, self.W_k.t()) 
      V = torch.matmul(x, self.W_v.t())

      Q_demod = Q /  sigma_j_q.reshape(batch_size, 1, -1)
      Q_demod = Q_demod.reshape(batch_size, flattened_im_len, num_heads, attn_depth).transpose(1,2)

      K_demod = K / sigma_j_k.reshape(batch_size, 1, -1)
      K_demod = K_demod.reshape(batch_size, flattened_im_len, num_heads, attn_depth).transpose(1,2)

      # Modulate Value with Style Value and save a residual for later use
      # See also the residual connection in a yellow box at fig.2(b)
      V_demod = V / sigma_j_v.reshape(batch_size, 1, -1)
      V_mod = V_demod * styles_val
      mod_residual = V_mod
      V_mod = V_mod.reshape(batch_size, flattened_im_len, num_heads, attn_depth).transpose(1,2)
      
      # Now we have the self-attention part
      attn = torch.matmul(Q_demod, K_demod.transpose(2,3))
      attention_probs = attn.softmax(dim=-1)
      weighted_values = torch.matmul(attention_probs, V_mod).transpose(1,2)
  
      x = weighted_values.reshape(batch_size, flattened_im_len, hidden_dimension) 
      x = torch.matmul(x, self.W_attn.t())

      # Demodulate Multihead Attention Integration, i.e. blue box in fig.2(b)
      # This part is named as "Attention Style Injection"
      # One thing that is unclear is that how do we demodulate the attention?
      # as attention takes demodulated Q and K but modulated V, we assume that
      # the demodulation is on the V side, which means we need to use Style Value
      # to demodulate the attention (the others Q and K are already demodulated)
      attn_mod_w = styles_val * self.W_attn.reshape(1, self.W_attn.shape[0], self.W_attn.shape[1])
      sigma_j_w = torch.sqrt(attn_mod_w.square().sum(dim=2))
      x = x / sigma_j_w.reshape(batch_size, 1, -1)
      
      res_mod_w = styles_val * self.W_residual.reshape(1, self.W_residual.shape[0], self.W_residual.shape[1])
      sigma_j_res = torch.sqrt(res_mod_w.square().sum(dim=2))
      
      # Quoted from the paper:
      # "... we perform linear operation to Mod Value, then perform demodulation operation
      # (same as demodulation for query, key, value)."
      #
      # Take the residual and modify it by passing through a linear layer
      modified_residual = torch.matmul(mod_residual, self.W_residual.t())
      # Demodulate the residual
      modified_residual = modified_residual / sigma_j_res.reshape(batch_size, 1, -1)

      # Add the Modified Residual (see yellow box in fig.2(b)) 
      # Note: Even though Modified Residual seems to come after Demod in the fig.2 (b)
      # paper mentions that "we perform linear operation to Mod Value, then perform demodulation operation"
      # Therefore, we first pass the residual connection to a linear layer, then perform demodulation
      x = x + modified_residual
      return x

