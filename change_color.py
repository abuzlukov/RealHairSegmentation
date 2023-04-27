import imghdr
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import gc
import ImportData as data_path
import matplotlib.pyplot as plt
import matplotlib.colors as clc
import os
import numpy as np
from skimage import color as skc


def change_color(img_file, mask_file,color):
    inp = str(input("What color do you want ? "))
    img_hsv = skc.rgb2hsv(img_file)
    color = skc.rgb2hsv(color[inp])
    print(color)
    img_hsv[:,:,0] = color[0]

    # assign the modified hue channel to hsv imag
    img_rgb = skc.hsv2rgb(img_hsv)
    plt.imshow(img_rgb)
    plt.pause(1)
    plt.show()
    img_file_rgb = skc.hsv2rgb(skc.rgb2hsv(img_file))
    img_out = img_file_rgb
    img_out = np.where(mask_file[:,:,0:3]==255,img_rgb,img_out)
    #print(img_file[0,0], img_out[0,0])
    plt.imshow(img_out)
    plt.pause(1)
    plt.show()

image_fp = os.path.join("..\..\Cours\S9\Projet_avance\published_DB\data_train\input", "frame00253.jpg")
img_file = plt.imread(image_fp)
maskl_fp = os.path.join("..\..\Cours\S9\Projet_avance\published_DB\data_train\ground", "frame00253.pbm")
mask_file = plt.imread(maskl_fp)
red = np.array([255.0,0.0,0.0])
green = np.array([0.0,255.0,0.0])
blue = np.array([0.0,0.0,150.0])
cyan = np.array([0.0,255.0,255.0])
yellow = np.array([255.0,255.0,0.0])
white = np.array([255.0,255.0,255.0])
black = np.array([0.0,0.0,0.0])

color = {'red':red, 'green':green, 'blue':blue, 'cyan':cyan, 'yellow':yellow, 'white': white, 'black':black}

change_color(img_file,mask_file,color)
