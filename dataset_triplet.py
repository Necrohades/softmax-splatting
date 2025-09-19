from torch.utils import data
from PIL import Image
import random
from pathlib import Path
from skimage import io
import numpy as np
import torch

import logging
log = logging.getLogger(__name__)

SEQ_LEN = 3
MAX_VALUE = 255.0

class Dataset(data.Dataset):
    def __init__(self, root: str, split: str, transforms=None, limit: int = -1, *args, **kwargs):
        """
        Creates a Vimeo Triplet object.
        Inputs.
            root: Root path for the Vimeo dataset containing the triplets.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.root = Path(root)
        self.image_root = self.root / "sequences"
        self.split = split
        self.transforms = transforms
        
        
        log.info(f"Loading {split} data from {self.image_root}")
        
        
        # Create the file path for the sequences list
        sequences_file = self.root / f"tri_{self.split}list.txt"
        
        log.info(f"Loading sequences from {sequences_file}")
        
        
        # Read the sequences from the file
        with open(sequences_file, 'r') as f:
            self.sequences = f.read().splitlines()
            
        # Limit the number of sequences if specified
        if limit > 0:
            log.info(f"Limiting the number of sequences to {limit}")
            self.sequences = self.sequences[:limit]
            
        log.info(f"Number of sequences available: {len(self.sequences)}")

    def __getitem__(self, index) -> tuple[list[torch.Tensor], torch.Tensor]:
        
        # Get the sequence directory
        sequence_dir = self.sequences[index]

        # Create image paths for the sequence
        imgpaths = [f"{self.image_root}/{sequence_dir}/im{i}.png" for i in range(1, SEQ_LEN+1)]

        # Randomly invert the sequence
        if random.random() >= 0.5:
            imgpaths = imgpaths[::-1]
        
        # Load images
        images = [io.imread(pth).astype(np.float32) / MAX_VALUE for pth in imgpaths]

        # Data augmentation
        if self.transforms:
            transformed = self.transforms(
                image=images[0],
                image1=images[1],
                image2=images[2]
            )
            images = [transformed['image'], transformed['image1'], transformed['image2']]

        # Split into input and ground truth
        mid_index = len(images) // 2
        gt = images[mid_index]
        images = images[:mid_index] + images[mid_index+1:]
        
        return images, gt

    def __len__(self):
        return len(self.sequences)
