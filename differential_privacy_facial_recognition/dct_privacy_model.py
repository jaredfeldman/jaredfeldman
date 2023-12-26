# Import the libraries we'll use below.
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import torch
from PIL import Image
import os
from IPython.display import display, HTML
import cv2
from torchvision import transforms, datasets
import torch.distributed as dist
from typing import List
import torch.nn.init as init
import torch.optim as optim
import torch.cpu.amp as amp
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from ignite.metrics import Accuracy, Precision, Recall
import torch.nn as nn
from PIL import Image, ImageEnhance

import tensorflow as tf
from tensorflow import keras
from keras import metrics
tf.get_logger().setLevel('INFO')
from util import ImageCNN, NoisyAccumulation
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchjpeg import dct
from sklearn import preprocessing
from ignite.utils import to_onehot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn
import scipy
from scipy.fftpack import idct
from tabulate import tabulate


"""
This file contains the majority of logic that handles:
- image/data loading
- dct transformation
- recognition model
- validation
"""

"""
Define the required inputs
"""

def arg_parse():
    parser = argparse.ArgumentParser(description='Train IRSE model with perturbed images based on DCT')
    parser.add_argument('--data-dir', help='path to images', required=True)
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train the model based on')
    parser.add_argument('--print-images', default=False, action='store_true', help='Just print images')

    args = parser.parse_args()
    return args

def target_to_oh(target):
    NUM_CLASS = 12  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot

"""
Used to setup ImageDataset, load in data, and create an iterable
"""
def prepare_data(image_dir, test_size=0.2):
    rgb_mean = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    rgb_std = [0.5, 0.5, 0.5]

    # series of transformations
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(), # apply some random flip for data augmentation
        transforms.ToTensor(), # convert the flipped image back to a pytorch tensor
        transforms.Normalize(mean=rgb_mean, std=rgb_std) # normalize the tensor using mean and std from above
    ])
    # setup the dataset
    data_dir = image_dir
    # train/test split
    train_data = datasets.ImageFolder(data_dir, transform, target_transform=target_to_oh)
    # Use DataLoader to create an iteratable object
    train_data_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    return train_data_loader, len(train_data)

"""
Helper IDCT function
"""
def block_idct(dct_block):
    """
    Apply Inverse Discrete Cosine Transform (IDCT) to each 8x8 block.
    Args:
    dct_block (numpy.ndarray): An array of DCT coefficients.
    Returns:
    numpy.ndarray: The reconstructed image blocks after IDCT.
    """
    # Define a function to apply IDCT to a single block
    def idct_2d(block):
        # Apply IDCT in both dimensions
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    # Assuming the input is of shape (bs, ch, h, w)
    bs, ch, h, w = dct_block.shape
    # Initialize an empty array for the output
    idct_image = np.zeros_like(dct_block, dtype=np.float32)
    # Apply IDCT to each block
    for b in range(bs):
        for c in range(ch):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    # Extract the block
                    block = dct_block[b, c, i:i+8, j:j+8]
                    # Perform IDCT
                    idct_image[b, c, i:i+8, j:j+8] = idct_2d(block)
    return idct_image

"""
This is where the logic to dct-fy images lives
"""
def images_to_batch(x):

    # input has range -1 to 1, this changes to range from 0 to 255
    x = (x + 1) / 2 * 255

    # scale_factor=8 does the blockify magic
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

    # raise error if # of channels does not equal 3
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return

    # convert to ycbcr
    x = dct.to_ycbcr(x)  # convert RGB to YCBCR

    # DCT is designed to work on values ranging from -128 to 127
    # Subtracting 128 from values 0-255 will change range to be -128 to 127
    # https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
    x -= 128

    # assign variables batch size, channel, height, weight based on x.shape
    bs, ch, h, w = x.shape

    # set the number of blocks
    block_num = h // 8
    # gives you insight of the stack that is fed into the "upsampling" piece
    x = x.view(bs * ch, 1, h, w)

    # 8 fold upsampling
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                 stride=(8, 8))

    # transposed to be able to feed into dct
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)

    # do dct
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    # remove DC as its important for visualization, but not recognition
    dct_block = dct_block[:, :, 1:, :, :]

    # Un DCT-fy it
    dc_coefficient = torch.zeros(bs, ch, 1, h // 8, w // 8, device=dct_block.device)
    dct_block = torch.cat((dc_coefficient, dct_block), dim=2)
    # Reshape to the format suitable for inverse DCT
    dct_block = dct_block.permute(0, 1, 3, 4, 2).reshape(bs, ch, h, w)
    # Apply inverse DCT
    x = dct.block_idct(dct_block)  # Assuming your dct module has a block_idct function
    # Add 128 to each pixel
    x += 128
    # Convert from YCbCr to RGB
    x = dct.to_rgb(x)  # Convert YCbCr back to RGB
    # Normalize the image back to the range -1 to 1
    x = F.interpolate(x, scale_factor=1/8, mode='bilinear', align_corners=True)
    x = (x / 255) * 2 - 1
    return x


def init_process_group():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

"""
Function used to compute the final accuracy, precision scores
"""
def compute_validation_score(model, test_data, noise_accumulation):

    for step, samples in enumerate(test_data):
        inputs = samples[0]
        labels = samples[1]
        # perturb the image
        inputs = images_to_batch(inputs)
        inputs = noise_accumulation(inputs)
        # get our r_pred
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        _, labels_updated = torch.max(labels, 1)
        # the above transforms the shapes of the predicted and labels from 773 x 12 -> 773

        precision = sklearn.metrics.precision_score(labels_updated, predicted, average="macro")
        accuracy = sklearn.metrics.accuracy_score(labels_updated, predicted)
        recall = sklearn.metrics.recall_score(labels_updated, predicted, average="macro")
        print(tabulate([[accuracy, precision, recall]], headers=['Model Accuracy', 'Model Precision', 'Model Recall']))

# TODO:
# Use the transforms to resize -- DONE
# Update how we're labelling to be the one-hot encoded version -- DONE
# Update the training portion of the model and replace it with the unperturbed images -- DONE
# Investigate why the perturbed image has different number of channels
# Run predictions against perturbed images -- DONE
# measure the accuracy, precision and recall -- DONE
def main(cnnModel, training_data, running_loss, e,):
    # setup noise object
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnnModel.parameters(), lr=0.001)
    train_losses = []

    # loop through each sample in the dataloader object
    for step, samples in enumerate(training_data):
        inputs = samples[0]
        labels = samples[1]

        # zero grad
        optimizer.zero_grad()
        output = cnnModel(inputs)
        loss = criterion(output, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()


    epoch_train_loss = running_loss / len(training_data.dataset)
    print('Epoch {}, train loss : {}'.format(e, epoch_train_loss))
    return cnnModel, epoch_train_loss

def printing_images(training_data, noise_accumulation):
    rgb_mean = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    rgb_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToPILImage(), # convert image to PIL for easier matplotlib reading
        transforms.RandomHorizontalFlip(), # apply some random flip for data augmentation
        transforms.ToTensor(), # convert the flipped image back to a pytorch tensor
        transforms.Normalize(mean=rgb_mean, std=rgb_std) # normalize the tensor using mean and std from above
    ])
    for step, samples in enumerate(training_data):
        inputs = samples[0]
        labels = samples[1]

        inputs = images_to_batch(inputs)
        inputs = noise_accumulation(inputs)

        # display the PIL Image
        for image in inputs:
            pil_img = transforms.ToPILImage()(image)

            plt.imshow(pil_img, interpolation="bicubic")
            plt.show()

def epoch_controller():
    # setup models and activation models required for model training and image generation
    args = arg_parse()
    #init_process_group()
    data_dir = args.data_dir
    num_epochs = args.epochs
    print_images = args.print_images

    training_data, num_samples = prepare_data(data_dir)
    noise_accumulation = NoisyAccumulation.NoisyAccumulation(budget_mean=4)

    if print_images:
        printing_images(training_data, noise_accumulation)
    else:
        print("Using this many images to train: ", len(training_data.dataset))

        save_epochs = [10, 18, 22, 24]
        losses = OrderedDict()

        noise_accumulation.train()
        running_loss = 0
        train_losses = []
        cnnModel = ImageCNN.ImageCNN()
        running_loss = 0
        for epoch in range(num_epochs):
            print("Training epoch: ", epoch)
            cnnModel, training_loss = main(cnnModel, training_data, running_loss, epoch)
            train_losses.append(training_loss)
            running_loss = training_loss

        print('Training losses:')
        print(train_losses)
        # compute validaion scores
        compute_validation_score(cnnModel, training_data, noise_accumulation)



if __name__ == "__main__":
    epoch_controller()
