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
MR-CT paired dataset for medical image synthesis.
Reads paired MR and CT images from directory structure:
    dataset/
        mr/
            train/
            test/
        ct/
            train/
            test/
Images are read in single-channel (grayscale) mode and normalized to (-1, 1).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MRCTDataset(Dataset):
    """
    Dataset for paired MR-CT images.
    
    Args:
        data_path: Root path containing 'mr' and 'ct' subdirectories
        split: 'train' or 'test'
        image_size: Target image size (default: 256)
        transform: Optional additional transforms
    """
    def __init__(self, data_path, split='train', image_size=256, transform=None):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Build paths
        self.mr_path = os.path.join(data_path, 'mr', split)
        self.ct_path = os.path.join(data_path, 'ct', split)
        
        # Get sorted file lists
        self.mr_files = sorted([f for f in os.listdir(self.mr_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.ct_files = sorted([f for f in os.listdir(self.ct_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Verify same number of files
        assert len(self.mr_files) == len(self.ct_files), \
            f"Number of MR files ({len(self.mr_files)}) != CT files ({len(self.ct_files)})"
        
        print(f"[MRCTDataset] Loaded {len(self.mr_files)} paired images from {split} split")
    
    def __len__(self):
        return len(self.mr_files)
    
    def _load_image(self, path):
        """Load image in grayscale mode and normalize to (-1, 1)"""
        img = Image.open(path).convert('L')  # Convert to grayscale
        
        # Resize if needed
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Convert to numpy array and normalize to (-1, 1)
        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np / 127.5) - 1.0  # [0, 255] -> (-1, 1)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img_np = img_np[np.newaxis, ...]
        
        return torch.from_numpy(img_np)
    
    def __getitem__(self, idx):
        # Load MR and CT images
        mr_path = os.path.join(self.mr_path, self.mr_files[idx])
        ct_path = os.path.join(self.ct_path, self.ct_files[idx])
        
        mr_img = self._load_image(mr_path)
        ct_img = self._load_image(ct_path)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            mr_img = self.transform(mr_img)
            ct_img = self.transform(ct_img)
        
        return mr_img, ct_img
    
    @staticmethod
    def denormalize(tensor):
        """
        Denormalize tensor from (-1, 1) to (0, 255) for saving.
        
        Args:
            tensor: Tensor with values in range (-1, 1)
            
        Returns:
            Tensor with values in range (0, 255) as uint8
        """
        # Clamp to (-1, 1) first
        tensor = torch.clamp(tensor, -1, 1)
        # Convert to (0, 255)
        tensor = (tensor + 1.0) * 127.5
        return tensor.to(torch.uint8)
