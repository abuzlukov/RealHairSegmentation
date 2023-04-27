#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 17:54:21 2022

@author: anass
"""
import argparse
from tensorflow.keras import  models
from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def main(model_directory,image_directory):

    model = models.load_model(model_directory)


    image = load(image_directory)

    plt.imshow(image.reshape((224,224)),cmap='gray')
    plt.show()


    pred=model.predict(image)
    plt.imshow(pred.reshape((224,224)),cmap='gray')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory',type=str,help='Path to model')
    parser.add_argument('image_directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()
    main(args.model_directory, args.image)