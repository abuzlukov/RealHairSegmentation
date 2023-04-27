# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:03:17 2022

@author: abdou
"""

import cv2   
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import os
# Size des données de bases et de ground truth 
x_path = 'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\Dataset_Hair\\Validation_folder\\input\\'
y_path = 'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\Dataset_Hair\\Validation_folder\\ground\\'
x_paths = os.listdir(x_path)
y_paths = os.listdir(y_path)
print("Count X:",len(x_paths))
print("Count Y:",len(y_paths))
images,masks = [],[]
# Réperage des élements qui ne suivent pas la structure
size = min(len(x_paths),len(y_paths))
for i in range(size):
    file = x_paths[i].replace('-org.jpg','')
    img_path,mask_path = file + '-org.jpg', file + '-gt.pbm'
    if img_path in x_paths and mask_path in y_paths:
        images.append( io.imread(x_path + img_path,plugin='matplotlib',as_gray = True) )
        masks.append( io.imread(y_path + mask_path,plugin='matplotlib',as_gray = True))
np_images = np.zeros((size,224,224,1))
np_masks = np.zeros((size,224,224,1))

# Normalisation des images pour avoir le même format 
for i in range(size):
    img = images[i]
    msk = masks[i]
    np_images[i] = resize(img,(224,224)).reshape((224,224,1))
    np_masks[i] = resize(msk,(224,224)).reshape((224,224,1))
    
    
    cv2.imwrite(f'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\New_Dataset_Hair\\Validation_folder\\input\\img_{i}.jpg', np_images[i])
    cv2.imwrite(f'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\New_Dataset_Hair\\Validation_folder\\ground\\mask_{i}.jpg', np_masks[i])


x_path = 'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\Dataset_Hair\\Training_folder\\input\\'
y_path = 'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\Dataset_Hair\\Training_folder\\ground\\'
x_paths = os.listdir(x_path)
y_paths = os.listdir(y_path)
print("Count X:",len(x_paths))
print("Count Y:",len(y_paths))
images,masks = [],[]
# Réperage des élements qui ne suivent pas la structure
size = min(len(x_paths),len(y_paths))
for i in range(size):
    file = x_paths[i].replace('-org.jpg','')
    img_path,mask_path = file + '-org.jpg', file + '-gt.pbm'
    if img_path in x_paths and mask_path in y_paths:
        images.append( io.imread(x_path + img_path,plugin='matplotlib',as_gray = True) )
        masks.append( io.imread(y_path + mask_path,plugin='matplotlib',as_gray = True))
np_images = np.zeros((size,224,224,1))
np_masks = np.zeros((size,224,224,1))

# Normalisation des images pour avoir le même format 
for i in range(size):
    img = images[i]
    msk = masks[i]
    np_images[i] = resize(img,(224,224)).reshape((224,224,1))
    np_masks[i] = resize(msk,(224,224)).reshape((224,224,1))
    
    
    cv2.imwrite(f'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\New_Dataset_Hair\\Training_folder\\input\\img_{i}.jpg', np_images[i])
    cv2.imwrite(f'C:\\Users\\abdou\\OneDrive - bordeaux-inp.fr\\Bureau\\3A\\PROJET S9\\Project\\New_Dataset_Hair\\Training_folder\\ground\\mask_{i}.jpg', np_masks[i])
