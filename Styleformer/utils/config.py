"""
CONFIG.PY

"""

# General dependencies
import os 
import math
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from typing import Iterator, Tuple
from torchvision.utils import make_grid
import torchvision.transforms as transforms

# Labml dependencies
from labml.configs import BaseConfigs
from labml_nn.utils import cycle_dataloader
from labml_helpers.device import DeviceConfigs
from labml import tracker, lab, monit, experiment
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.stylegan import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss

# Source file dependencies
from models import generator, discriminator

"""
Configs Class
**Disclaimer:** This class is based on LABML's StyleGAN2 implementation. See [their experiment.py](https://nn.labml.ai/gan/stylegan/experiment.html) implementation. 

We have changed some parameters and added some blocks to save the model outputs, to fit in our case.
See the comments with "Edit:" to check out our additions. We also commented out or deleted some parts that are not used in Styleformer paper (such as PathlengthPenalty)
"""

class Configs(BaseConfigs):  
    
    if not torch.cuda.is_available():
        device: torch.device = torch.device("cpu")
            
    else:
        device: torch.device = DeviceConfigs()
            
    """
    # Commented here because "mps" device is not working properly with pytorch gradients
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("GPU support is available and enabled!")
        device: torch.device = torch.device("mps")
    else:
        device: torch.device = DeviceConfigs()
    """
    
    discriminator: discriminator.Discriminator
    generator: generator.Generator

    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss

    # Optimizers
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam

    gradient_penalty = GradientPenalty()
    gradient_penalty_coefficient: float = 10.

    path_length_penalty: PathLengthPenalty

    # Data loader
    loader: Iterator

    # Batch size
    batch_size: int = 32
    # Dimensionality of $z$ and $w$
    # d_latent: int = 512 # see generator parameters
    
    # Height/width of the image
    image_size: int = 32
    
    # Generator & Discriminator learning rate
    learning_rate: float = 1e-3
    
    # Number of steps to accumulate gradients on. Use this to increase the effective batch size.
    gradient_accumulate_steps: int = 1
    # $\beta_1$ and $\beta_2$ for Adam optimizer
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    # Probability of mixing stylesZ
    is_style_mixing: bool = False     # Set to False for CIFAR10, Styleformer Appendix.A
    style_mixing_prob: float = 0.9

    # Total number of training steps
    training_steps: int = 150_000

    #--- Generator Parameters --------------------------------------------------
    h_dim_w:          int = 256
    w_depth:          int = 2  # This is only for CIFAR10, see Styleformer Appendix.A
    im_resolution:    int = 32
    attn_depth:       int = 32
    num_enc_blocks:   list[int] = [1,2,2]       # See Styleformer Table 5 Layers {1, 3, 3}
    enc_dims:         list[int] = [256,64,16]   # See Styleformer Table 5 Hidden Size {1024,512,512}
    #---------------------------------------------------------------------------

    #--- Discriminator Parameters ----------------------------------------------
    channel_base:        int = 2**11    # Overall multiplier for the number of channels.
    channel_max:         int = 512      # Maximum number of channels in any layer.
    last_resolution:     int = 4        # Resolution of the last layer
    mbstd_group_size:    int = 4        # Group size for the minibatch standard deviation layer
    #---------------------------------------------------------------------------

    # Number of blocks in the generator (calculated based on image resolution)
    n_gen_blocks: int

    # ### Lazy regularization
    # Instead of calculating the regularization losses, the paper proposes lazy regularization
    # where the regularization terms are calculated once in a while.
    # This improves the training efficiency a lot.

    # The interval at which to compute gradient penalty
    lazy_gradient_penalty_interval: int = 4
    # Path length penalty calculation interval
    lazy_path_penalty_interval: int = 32
    # Skip calculating path length penalty during the initial phase of training
    lazy_path_penalty_after: int = 5_000

    # How often to log generated images
    log_generated_interval: int = 500
    # How often to save model checkpoints
    save_checkpoint_interval: int = 2_000

    # Training mode state for logging activations
    mode: ModeState
    # Whether to log model layer outputs
    log_layer_outputs: bool = False
    
    # Edit: We have added another path to save the necessary files
    # Edit v.2: We enable specifying pretrained model for generator and discriminator seperately
    # such that we can only load generator, and train discriminator from scratch.
    path_to_save: str 
    path_gen_model: str 
    path_disc_model: str 
    path_loss_plot: str 
    starting_step: int = 0
        

    def init(self, dataloader = None):
        """
        ### Initialize
        """
      
        self.loader = cycle_dataloader(dataloader)

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.image_size))

        # Create discriminator and generator
        self.discriminator = discriminator.Discriminator(
            self.im_resolution,
            self.channel_base,
            self.channel_max,
            self.last_resolution,
            self.mbstd_group_size,
        ).to(self.device)

        self.generator = generator.Generator(
            self.h_dim_w,
            self.w_depth,
            self.im_resolution,
            self.attn_depth,
            self.num_enc_blocks,
            self.enc_dims,
        ).to(self.device)

        if self.starting_step: 
            if torch.cuda.is_available():
                self.generator.load_state_dict(torch.load(self.path_gen_model))
                self.discriminator.load_state_dict(torch.load(self.path_disc_model))
            else:
                self.generator.load_state_dict(torch.load(self.path_gen_model, map_location=torch.device('cpu')))
                self.discriminator.load_state_dict(torch.load(self.path_disc_model, map_location=torch.device('cpu')))

        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = len(self.num_enc_blocks)

        # Add model hooks to monitor layer outputs
        if self.log_layer_outputs:
            hook_model_outputs(self.mode, self.discriminator, 'discriminator')
            hook_model_outputs(self.mode, self.generator, 'generator')

        # Discriminator and generator losses
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        # Create optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )

        # Set tracker configurations
        tracker.set_image("generated", True)

    def generate_images(self, batch_size: int):   
   
        z = torch.randn(batch_size, self.h_dim_w).to(self.device) 
        return self.generator(z)

    def step(self, idx: int):
        
        # Train the discriminator
        with monit.section('Discriminator'):
            # Reset gradients
            self.discriminator_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Update `mode`. Set whether to log activation
                with self.mode.update(is_log_activations=(idx + 1) % self.log_generated_interval == 0):
                    # Sample images from generator
                    generated_images = self.generate_images(self.batch_size)

                    # Discriminator classification for generated images
                    fake_output = self.discriminator(generated_images.detach())

                    # Get real images from the data loader
                    real_images = next(self.loader).to(self.device)
                  
                    # We need to calculate gradients w.r.t. real images for gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()
                    # Discriminator classification for real images
                    real_output = self.discriminator(real_images)

                    # Get discriminator loss
                    real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                    disc_loss = real_loss + fake_loss

                    # Add gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        # Calculate and log gradient penalty
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add('loss.gp', gp)
                        # Multiply by coefficient and add gradient penalty
                        disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

                    # Compute gradients
                    disc_loss.backward()
                    # Log discriminator loss
                    tracker.add('loss.discriminator', disc_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add('discriminator', self.discriminator)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            # Take optimizer step
            self.discriminator_optimizer.step()

        # Train the generator
        with monit.section('Generator'):
            # Reset gradients
            self.generator_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Sample images from generator
                generated_images = self.generate_images(self.batch_size)

                # Discriminator classification for generated images
                fake_output = self.discriminator(generated_images)

                # Get generator loss
                gen_loss = self.generator_loss(fake_output)

                # Add path length penalty
                # Edit: Disabled path length penalty for Cifar10 as mentioned in the paper
               
                # Calculate gradients
                gen_loss.backward()
                # Log generator loss
                tracker.add('loss.generator', gen_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add('generator', self.generator)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

            # Take optimizer step
            self.generator_optimizer.step()

        # Log generated images
        if (idx + 1) % self.log_generated_interval == 0:
            tracker.add('generated', generated_images[:9])
            # Edit: We have added another block of code to save the generated images as a grid
            imgs_gen = self.generate_images(128)
            plt.figure(figsize=(16,8))
            plt.axis('off')
            plt.title(f'generated_images_{idx + self.starting_step}')
            img_gen_denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))(imgs_gen.detach().cpu())  
            plt.imshow(make_grid((img_gen_denorm * 255).to(dtype=torch.uint8), nrow=16).permute((1, 2, 0)))
            plt.savefig(os.path.join(self.path_to_save, f'generated_images_{idx + self.starting_step}.png'), bbox_inches='tight')      

        # Save model checkpoints
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()
        # Flush tracker
        tracker.save()
        return gen_loss.detach().cpu(), disc_loss.detach().cpu()

    def train(self):

        g_loss_list = []
        d_loss_list = []      
        g_list = []
        d_list = []
      
        # Loop for `training_steps`
        for i in monit.loop(self.training_steps):
            # self.step(i)
            # Edit record the loss data of each step
            g_loss, d_loss = self.step(i)
            g_list.append(g_loss)
            d_list.append(d_loss)       

            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()
                # Edit: Take the average loss for each generated image interval
                # This reduces the spikes in the loss plot (where high variation
                # lose details) hence we are able to observe the behaviour better               
                g_loss_list.append(np.mean(np.array(g_list)))
                d_loss_list.append(np.mean(np.array(d_list)))
                
                # Edit: Plot the smoothed loss graphs
                plt.plot(g_loss_list, label="g_loss")
                plt.plot(d_loss_list, label="d_loss")                 
                plt.legend()
                plt.savefig(f'{self.path_loss_plot}/loss_plot_{i}.png')
                plt.show()
                
                g_list = []
                d_list = []
