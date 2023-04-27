# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:20:23 2022

@author: anass
"""
import torch as T


import os
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import torch as nn
import torch
from PIL import Image

import torchvision 
import torch
import torchvision.transforms as transforms

from torchvision.datasets import VisionDataset


class ImportData(VisionDataset):

    
    def __init__(
        self,
        root: str,
    ):
        super().__init__(
            root,
        )
        self.images_input = []
        self.images_ground = []
        self.root='/home/anass/OPPPYTHON/RealHairSegmentation/Dataset_Hair/training/'
        image_dir_input = os.path.join(self.root, 'input/')
        image_dir_ground = os.path.join(self.root, 'ground/')

        for img_file in os.listdir(image_dir_input):
            self.images_input.append(os.path.join(image_dir_input, img_file))
        for img_file in os.listdir(image_dir_ground):
            self.images_ground.append(os.path.join(image_dir_ground, img_file))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:
     """
        img=Image.open(self.images_input[index]).convert("L")
        ground=Image.open(self.images_ground[index]).convert("L")
        transform_image = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])(img)
        transform_ground = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])(ground)
        return [transform_image, transform_ground]
    def __len__(self) -> int:
        return len(self.images_input)