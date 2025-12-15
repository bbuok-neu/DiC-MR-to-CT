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
Sampling/inference script for MR-to-CT medical image synthesis using DiC.
Takes MR images as input and generates corresponding CT images.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os

from diffusion import create_diffusion
from utils.mrct_dataset import MRCTDataset


def main(args, unparsed):
    """
    Run MR-to-CT sampling/inference.
    """
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup DDP if available
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception:
        rank = 0
        world_size = 1

    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    opts = dict()
    from utils.parser_setter import extract_parser, printopt
    extract_parser(unparsed, opts)
    if rank == 0:
        print('----> Opt printed as follows:')
        printopt(opts)

    # Import DiC model
    from dic_models import DiC_models

    # Load model
    assert args.ckpt is not None, "Must specify --ckpt for sampling"
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt_args = ckpt.get('args', None)
    
    # Determine learn_sigma from checkpoint if not specified
    learn_sigma = args.learn_sigma
    if ckpt_args is not None and hasattr(ckpt_args, 'learn_sigma'):
        learn_sigma = ckpt_args.learn_sigma
        if rank == 0:
            print(f"Using learn_sigma={learn_sigma} from checkpoint")
    
    model = DiC_models[args.model](
        input_size=args.image_size,
        in_channels=2,  # 1 channel noisy CT + 1 channel MR condition
        num_classes=1,
        learn_sigma=learn_sigma,
        **(opts['network_g'] if opts.get('network_g') is not None else dict())
    ).to(device)
    
    # Load weights - try EMA first, then model
    if args.use_ema and 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        if rank == 0:
            print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model'])
        if rank == 0:
            print("Loaded model weights")
    
    model.eval()
    if rank == 0:
        print(f"Loaded checkpoint from {args.ckpt}")

    # Create diffusion for sampling
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=learn_sigma)

    # Create output directory
    ckpt_name = os.path.basename(args.ckpt).replace(".pt", "")
    output_dir = os.path.join(
        args.output_dir,
        f"{args.model}-{ckpt_name}-steps{args.num_sampling_steps}-seed{args.global_seed}"
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ct_pred"), exist_ok=True)
        if args.save_comparison:
            os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
        print(f"Saving results to {output_dir}")
    
    if world_size > 1:
        dist.barrier()

    # Load test dataset
    test_dataset = MRCTDataset(args.data_path, split='test', image_size=args.image_size)
    
    if world_size > 1:
        sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    if rank == 0:
        print(f"Test dataset contains {len(test_dataset)} paired images")

    # Run sampling
    sample_idx = 0
    pbar = tqdm(test_loader, disable=(rank != 0))
    
    for mr, ct_gt in pbar:
        mr = mr.to(device)
        ct_gt = ct_gt.to(device)
        batch_size = mr.shape[0]
        
        # Generate initial noise
        z = torch.randn(batch_size, 1, args.image_size, args.image_size, device=device)
        
        # Dummy class label
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Define model function that concatenates noise and MR condition
        def model_fn(x, t, y):
            # Concatenate noisy CT and MR condition
            x_concat = torch.cat([x, mr], dim=1)  # (B, 2, H, W)
            out = model(x_concat, t, y)
            # Model outputs 2 channels (or 4 if learn_sigma), but we only need CT channels
            if learn_sigma:
                # Return noise and variance for CT (channels 0 and 2)
                return torch.cat([out[:, :1, :, :], out[:, 2:3, :, :]], dim=1)
            else:
                return out[:, :1, :, :]  # Only first channel for CT noise prediction
        
        # Sample using DDPM
        samples = diffusion.p_sample_loop(
            model_fn, z.shape, z, clip_denoised=True,
            model_kwargs=dict(y=y), progress=False, device=device
        )
        
        # Denormalize and save results
        for i in range(batch_size):
            global_idx = sample_idx + i
            if world_size > 1:
                global_idx = global_idx * world_size + rank
            
            # Denormalize CT prediction
            ct_pred = MRCTDataset.denormalize(samples[i].squeeze(0).cpu())
            ct_pred_np = ct_pred.numpy()
            
            # Save CT prediction
            ct_pred_img = Image.fromarray(ct_pred_np)
            ct_pred_img.save(os.path.join(output_dir, "ct_pred", f"{global_idx:06d}.png"))
            
            # Optionally save comparison (MR | CT_GT | CT_Pred)
            if args.save_comparison:
                mr_np = MRCTDataset.denormalize(mr[i].squeeze(0).cpu()).numpy()
                ct_gt_np = MRCTDataset.denormalize(ct_gt[i].squeeze(0).cpu()).numpy()
                comparison = np.concatenate([mr_np, ct_gt_np, ct_pred_np], axis=1)
                comparison_img = Image.fromarray(comparison)
                comparison_img.save(os.path.join(output_dir, "comparison", f"{global_idx:06d}.png"))
        
        sample_idx += batch_size

    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"Done! Generated {sample_idx} samples.")
        print(f"Results saved to {output_dir}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset directory containing 'mr' and 'ct' subdirs")
    parser.add_argument("--output-dir", type=str, default="samples_mrct",
                        help="Directory to save generated samples")
    parser.add_argument("--image-size", type=int, default=256)
    
    # Model settings
    parser.add_argument("--model", type=str, default="DiC-XL")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights for sampling (default: True)")
    parser.add_argument("--learn-sigma", action="store_true",
                        help="Model learns variance (will be auto-detected from checkpoint)")
    
    # Sampling settings
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for sampling")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    
    # Output settings
    parser.add_argument("--save-comparison", action="store_true", default=True,
                        help="Save MR|CT_GT|CT_Pred comparison images")
    
    args, unparsed = parser.parse_known_args()
    main(args, unparsed)
