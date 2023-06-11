import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from models import Generator, Discriminator, MaskGenerator
from dataset import load_data
from train import train
from utils import (
    set_logger,
    set_fix_seed,
    save_model,
    load_model,
    generate_results,
    show_image_grid,
    print_metrics,
    save_generated_images,
    load_tracer_masks,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FurryGAN")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_file_path", type=str, default="./")
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--mode", type=str, default="train")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--max_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--style_dim", type=int, default=128)
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--reg_every", type=int, default=16)

    parser.add_argument("--gamma_start", type=float, default=0.0)
    parser.add_argument("--gamma_max", type=float, default=1.0)
    parser.add_argument("--gamma_step", type=float, default=0.01)
    parser.add_argument("--phi1", type=float, default=0.35)
    parser.add_argument("--phi2", type=float, default=0.01)
    parser.add_argument("--lambda_coarse", type=float, default=5.0)
    parser.add_argument("--lambda_fine", type=float, default=5.0)
    parser.add_argument("--lambda_min", type=float, default=0.5)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--r1", type=float, default=10, help="R1 regularization weight")

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time = datetime.now()
    log_file = (
        f"train_{time.hour}:{time.minute}_{time.day}-{time.month}-{time.year}.log"
    )
    args.logger = set_logger(os.path.join(args.log_file_path, log_file))
    set_fix_seed(0)

    train_loader, test_loader = load_data(args)

    fg_generator = Generator(args=args, mode="foreground").to(args.device)
    bg_generator = Generator(args=args, mode="background").to(args.device)
    mask_generator = MaskGenerator(args=args).to(args.device)
    discriminator = Discriminator(args=args).to(args.device)

    regulation_ratio = args.reg_every / (args.reg_every + 1)
    generator_params = (
        list(fg_generator.parameters())
        + list(bg_generator.parameters())
        + list(mask_generator.parameters())
    )
    generator_optim = optim.Adam(
        generator_params,
        lr=args.lr,
        betas=(0.0, 0.99),
    )
    discriminator_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * regulation_ratio * 0.9,
        betas=(0.0, 0.99**regulation_ratio),
    )

    models = [fg_generator, mask_generator, bg_generator]
    save_paths = [
        os.path.join(args.model_save_path, "fg_generator.pt"),
        os.path.join(args.model_save_path, "mask_generator.pt"),
        os.path.join(args.model_save_path, "bg_generator.pt"),
    ]
    if args.mode == "train":
        train(
            fg_generator,
            mask_generator,
            bg_generator,
            discriminator,
            generator_optim,
            discriminator_optim,
            train_loader,
            args,
        )
        save_model(args, models, save_paths)
    elif args.mode == "test":
        load_model(models, save_paths)

    # Qualitative results
    fg_images, masks, bg_images, comp_images = generate_results(
        args, fg_generator, mask_generator, bg_generator, sample_size=2
    )
    show_image_grid(fg_images, masks, bg_images, comp_images, -1)

    # Quantitative results
    fg_images, masks, bg_images, comp_images = generate_results(
        args, fg_generator, mask_generator, bg_generator, sample_size=args.batch_size
    )
    save_generated_images(comp_images)
    os.system("python ./TRACER/main.py")
    gt_fg_masks = load_tracer_masks(args.batch_size)
    gt_bg_masks = 1 - gt_fg_masks

    real_images = next(iter(test_loader))[0].to(args.device)
    fg_masks = masks
    bg_masks = 1 - masks
    print_metrics(
        real_images, comp_images, gt_fg_masks, fg_masks, gt_bg_masks, bg_masks, args.device
    )
