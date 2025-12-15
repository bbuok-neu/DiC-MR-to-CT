# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Training script for MR-to-CT medical image synthesis using DiC.
Uses 2-channel input by concatenating noisy CT (1 ch) and MR condition (1 ch).
Supports:
- Resume training from checkpoint
- Online validation during training
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from diffusion import create_diffusion
from utils.mrct_dataset import MRCTDataset
from utils.parser_setter import extract_parser, printopt
from dic_models import DiC_models


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir, rank=0):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def validate(model, diffusion, val_loader, device, num_samples=4, num_sampling_steps=250, learn_sigma=False):
    """
    Run validation by generating CT samples from MR images.
    
    Args:
        model: The DiC model (or EMA model)
        diffusion: Diffusion object for sampling
        val_loader: Validation data loader
        device: Device to run on
        num_samples: Number of samples to generate for validation
        num_sampling_steps: Number of diffusion sampling steps
        learn_sigma: Whether model learns variance
    
    Returns:
        List of (mr, ct_gt, ct_pred) tuples
    """
    model.eval()
    results = []
    
    # Create sampling diffusion with specified steps
    sample_diffusion = create_diffusion(str(num_sampling_steps), learn_sigma=learn_sigma)
    
    samples_collected = 0
    for mr, ct_gt in val_loader:
        if samples_collected >= num_samples:
            break
            
        mr = mr.to(device)
        ct_gt = ct_gt.to(device)
        batch_size = mr.shape[0]
        
        # Take only what we need
        remaining = num_samples - samples_collected
        if batch_size > remaining:
            mr = mr[:remaining]
            ct_gt = ct_gt[:remaining]
            batch_size = remaining
        
        # Generate noisy CT latent (start from pure noise)
        # For MR-CT task, we work in image space (single channel), not latent space
        latent_size = mr.shape[-1]
        z = torch.randn(batch_size, 1, latent_size, latent_size, device=device)
        
        # Dummy class label (not used in MR-CT conditioning)
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Sample CT from noise conditioned on MR
        def model_fn(x, t, y):
            # Concatenate noisy CT and MR condition
            x_concat = torch.cat([x, mr], dim=1)  # (B, 2, H, W)
            out = model(x_concat, t, y)
            # Model outputs 2 channels (or 4 if learn_sigma), but we only need CT channels
            # For learn_sigma=False: model outputs 2 ch, return first 1 ch for CT noise
            # For learn_sigma=True: model outputs 4 ch, return first 2 ch (1 noise + 1 variance) for CT
            if learn_sigma:
                # Return noise and variance for CT (channels 0 and 2)
                return torch.cat([out[:, :1, :, :], out[:, 2:3, :, :]], dim=1)
            else:
                return out[:, :1, :, :]  # Only first channel for CT noise prediction
        
        # Use DDPM sampling
        samples = sample_diffusion.p_sample_loop(
            model_fn, z.shape, z, clip_denoised=True,
            model_kwargs=dict(y=y), progress=False, device=device
        )
        
        # Store results
        for i in range(batch_size):
            results.append((
                mr[i].cpu(),
                ct_gt[i].cpu(),
                samples[i].cpu()
            ))
        
        samples_collected += batch_size
    
    model.train()
    return results


def save_validation_images(results, save_dir, step, denormalize_fn):
    """
    Save validation results as images.
    
    Args:
        results: List of (mr, ct_gt, ct_pred) tuples
        save_dir: Directory to save images
        step: Current training step
        denormalize_fn: Function to denormalize images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, (mr, ct_gt, ct_pred) in enumerate(results):
        # Denormalize
        mr_img = denormalize_fn(mr.squeeze(0)).numpy()
        ct_gt_img = denormalize_fn(ct_gt.squeeze(0)).numpy()
        ct_pred_img = denormalize_fn(ct_pred.squeeze(0)).numpy()
        
        # Create side-by-side comparison
        comparison = np.concatenate([mr_img, ct_gt_img, ct_pred_img], axis=1)
        
        # Save
        img = Image.fromarray(comparison)
        img.save(os.path.join(save_dir, f"step{step:07d}_sample{idx:02d}.png"))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, unparsed):
    """
    Trains DiC model for MR-to-CT synthesis.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
    except Exception:
        world_size = 1

    assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    opts = dict()
    extract_parser(unparsed, opts)
    if rank == 0:
        print('----> Opt printed as follows:')
        printopt(opts)

    # Setup experiment folder or resume from existing
    if args.resume:
        # Resume from existing checkpoint directory
        assert args.ckpt is not None, "Must specify --ckpt when using --resume"
        checkpoint_dir = os.path.dirname(args.ckpt)
        experiment_dir = os.path.dirname(checkpoint_dir)
        if rank == 0:
            logger = create_logger(experiment_dir, rank)
            logger.info(f"Resuming training from {args.ckpt}")
    else:
        # Create new experiment folder
        if rank == 0:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-MRCT"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir, rank)
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            logger = create_logger(None, rank)
    
    # Broadcast experiment_dir and checkpoint_dir to all ranks
    if world_size > 1:
        if rank == 0:
            exp_dir_list = [experiment_dir, checkpoint_dir]
        else:
            exp_dir_list = [None, None]
        dist.broadcast_object_list(exp_dir_list, src=0)
        experiment_dir, checkpoint_dir = exp_dir_list

    # Create model with 2-channel input (noisy CT + MR condition)
    # Output is 1 channel (or 2 if learn_sigma=True for CT prediction)
    model = DiC_models[args.model](
        input_size=args.image_size,  # Image size directly (no VAE encoding)
        in_channels=2,  # 1 channel noisy CT + 1 channel MR condition
        num_classes=1,  # No class conditioning for MR-CT task
        learn_sigma=args.learn_sigma,
        **(opts['network_g'] if opts.get('network_g') is not None else dict())
    )
    if rank == 0:
        print(model)
        logger.info(f"DiC Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint if specified
    train_steps = 0
    if args.ckpt is not None:
        if rank == 0:
            print(f'Loading checkpoint {args.ckpt}')
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        train_steps = ckpt.get('train_steps', 0)
        if rank == 0:
            logger.info(f"Loaded checkpoint at step {train_steps}")

    # Setup EMA and DDP
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device]) if dist.is_initialized() else model.to(device)
    
    # Load EMA state if resuming
    if args.ckpt is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    else:
        update_ema(ema, model.module if world_size > 1 else model, decay=0)
    
    # Create diffusion
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=args.learn_sigma)

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Load optimizer state if resuming
    if args.ckpt is not None and 'opt' in ckpt:
        opt.load_state_dict(ckpt['opt'])
        del ckpt
        torch.cuda.empty_cache()

    # Setup data
    train_dataset = MRCTDataset(args.data_path, split='train', image_size=args.image_size)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if rank == 0:
        logger.info(f"Training dataset contains {len(train_dataset):,} paired images ({args.data_path})")

    # Setup validation data if enabled
    val_loader = None
    if args.validate:
        val_dataset = MRCTDataset(args.data_path, split='test', image_size=args.image_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        if rank == 0:
            logger.info(f"Validation dataset contains {len(val_dataset):,} paired images")
            os.makedirs(f"{experiment_dir}/validation", exist_ok=True)

    # Prepare models for training
    model.train()
    ema.eval()

    # Variables for monitoring/logging
    log_steps = 0
    running_loss = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
    
    epoch_start = train_steps * world_size * int(args.global_batch_size // world_size) // len(train_dataset)
    
    for epoch in range(epoch_start, args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        
        for mr, ct in train_loader:
            mr = mr.to(device)
            ct = ct.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (ct.shape[0],), device=device)
            
            # Create noisy CT (x_t) from clean CT (x_0)
            noise = torch.randn_like(ct)
            ct_noisy = diffusion.q_sample(ct, t, noise=noise)
            
            # Concatenate noisy CT and MR condition as input
            x_input = torch.cat([ct_noisy, mr], dim=1)  # (B, 2, H, W)
            
            # Dummy class label (not used)
            y = torch.zeros(ct.shape[0], dtype=torch.long, device=device)
            
            # Forward pass - model predicts noise (and optionally variance)
            model_output = model(x_input, t, y)
            
            # Compute loss
            # Model outputs 2 channels (same as input), but CT is only 1 channel
            # For learn_sigma=False: output has 2 channels, use first for CT noise
            # For learn_sigma=True: output has 4 channels, split into 2 for noise and 2 for variance
            B = ct.shape[0]
            if args.learn_sigma:
                # Output is 4 channels: 2 for noise (CT+MR), 2 for variance
                # We only use first channel for CT noise prediction
                noise_pred = model_output[:, :1, :, :]  # First channel only
            else:
                # Output is 2 channels, use first for CT noise prediction
                noise_pred = model_output[:, :1, :, :]
            
            # MSE loss between predicted noise and actual noise
            mse_loss = torch.mean((noise_pred - noise) ** 2)
            
            # Total loss
            loss = mse_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module if world_size > 1 else model)

            # Log loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if world_size > 1:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if world_size > 1:
                    dist.barrier()

            # Validation
            if args.validate and train_steps % args.val_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Running validation at step {train_steps}...")
                    results = validate(
                        ema, diffusion, val_loader, device,
                        num_samples=args.val_num_samples,
                        num_sampling_steps=args.val_sampling_steps,
                        learn_sigma=args.learn_sigma
                    )
                    save_validation_images(
                        results,
                        f"{experiment_dir}/validation",
                        train_steps,
                        MRCTDataset.denormalize
                    )
                    logger.info(f"Saved {len(results)} validation samples")
                if world_size > 1:
                    dist.barrier()

    model.eval()
    if rank == 0:
        logger.info("Training complete!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset directory containing 'mr' and 'ct' subdirs")
    parser.add_argument("--results-dir", type=str, default="results_mrct")
    parser.add_argument("--image-size", type=int, default=256)
    
    # Model settings
    parser.add_argument("--model", type=str, default="DiC-XL")
    parser.add_argument("--learn-sigma", action="store_true",
                        help="Learn variance in addition to mean")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    
    # Resume training
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint (saves to same directory)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint for resuming or loading weights")
    
    # Validation settings
    parser.add_argument("--validate", action="store_true",
                        help="Enable online validation during training")
    parser.add_argument("--val-every", type=int, default=5000,
                        help="Validation frequency (steps)")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-num-samples", type=int, default=4,
                        help="Number of validation samples to generate")
    parser.add_argument("--val-sampling-steps", type=int, default=250,
                        help="Number of diffusion steps for validation sampling")
    
    args, unparsed = parser.parse_known_args()
    main(args, unparsed)
