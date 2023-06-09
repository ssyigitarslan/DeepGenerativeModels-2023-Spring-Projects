import argparse
import importlib
import json
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
import numpy as np
from utils import label2rgb, get_pt_color, penalty
import os
from model import Generator, Discriminator
from tqdm import tqdm
from Parameters import Params
from evaluate import evaluate
from dataset import CelebAWildTrain


def train():
    args = Params()
    args.log = "GanSEG"

    os.makedirs(args.log, exist_ok=True)
    with open(os.path.join(args.log, 'parameters.json'), 'wt') as f:
        json.dump(args.__dict__, f, indent=2)

    device = 'cuda:0'
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    generator = Generator(args).to(device)
    discriminator = Discriminator().to(device)
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.9))
    optim_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(0.5, 0.9))

    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)
    dataset = CelebAWildTrain(args.data_root, args.image_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                        for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
    test_input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
    checkpoint_dir = os.path.join(args.log, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)


    for epoch in range(200):    
        discriminator.train()
        generator.train()
        total_disc_loss = 0
        total_gen_loss = 0

        for batch_index, real_batch in tqdm(enumerate(data_loader)):
            optim_disc.zero_grad()
            optim_gen.zero_grad()

            # update discriminator
            real_batch = {'img': real_batch['img'].to(device)}
            real_batch['img'].requires_grad_()
            input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                        for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
            input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
            fake_batch = generator(input_batch)
            d_real_out = discriminator(real_batch)
            d_fake_out = discriminator(fake_batch)
            disc_loss = F.softplus(d_fake_out).mean() + F.softplus(-d_real_out).mean()
            # print(disc_loss)
            disc_loss.backward()
            total_disc_loss += disc_loss.item()
            optim_disc.step()

            optim_disc.zero_grad()
            optim_gen.zero_grad()
            input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                            for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
            input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
            fake_batch = generator(input_batch,requires_penalty=True)
            d_fake_out = discriminator(fake_batch)
            gen_loss = F.softplus(-d_fake_out).mean()
            if batch_index % 2 == 0:
                gen_loss = gen_loss + fake_batch['center_penalty'].mean() * args.con_penalty_coef + fake_batch['area_penalty'].mean() * args.area_penalty_coef
            gen_loss.backward()
            total_gen_loss += gen_loss.item()
            optim_gen.step()
            disc_loss, gen_loss = total_disc_loss / len(data_loader) / 2, total_gen_loss / len(data_loader)
            print(f"Epoch: {epoch + 1}, disc_loss:  {disc_loss}, gen_loss: {gen_loss}")

        evaluate(generator, test_input_batch, args, epoch)

        if (epoch + 1) % 1 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'optim_gen': optim_gen.state_dict(),
                    'optim_disc': optim_disc.state_dict(),
                },
                os.path.join(checkpoint_dir, 'epoch_{}.model'.format(epoch))
            )


if __name__ == "__main__":
    train()
