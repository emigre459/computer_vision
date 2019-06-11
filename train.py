# -------------------- IMPORT PACKAGES --------------------

import argparse
import os
from copy import deepcopy
import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


# -------------------- SETUP MAJOR INPUT VALUES --------------------

# Setup major inputs required from command line
parser = argparse.ArgumentParser(description='Trains a neural network')
parser.add_argument('data_directory', type=str, 
    help='Filepath for input data. Expected to be the parent directory with \
    folders train, validation, and test inside, with each structured \
    according to torchivision.datasets.ImageFolder requirements')

# Setup optional parameters that can be entered from the command line
parser.add_argument('-s', '--save_dir', type=str, 
    default = 'model_checkpoints/', 
    help = 'Filepath indicating where trained model checkpoint files \
    should be stored')

parser.add_argument('-a', '--arch', type=str, 
    default = 'inception', 
    help = 'Pre-trained model from torchivision.models to use \
    for the feature detector layers of your model')

parser.add_argument('-l', '--learning_rate', type=float, 
    default = 0.0005, 
    help = 'Learning rate to use for the Adam optimizer')

parser.add_argument('-u', '--hidden_units', type=list, 
    default = [512, 256], 
    help = 'Number of nodes to use in each hidden layer, ordered from \
    earliest to latest layer. Not inclusive of the input layer \
    (node count dictated by model architecture chosen) and \
    output layer (always 102 = number of flower labels)')

parser.add_argument('-d', '--dropout', type=bool, 
    default = True, 
    help = 'Determines if dropout with p=0.2 will be used for \
    each hidden layer')


args = parser.parse_args()


# -------------------- ARCHITECTURE-SPECIFIC SETUP --------------------
# Sets parameters for various items that are not model-architecture-agnostic

# torchvision.models.inception_v3()
if args.arch == 'inception':
    crop_size = 299
    models.inception_v3(pretrained=True)
    classifier = model.fc
    input_nodes = 2048


elif args.arch == 'densenet': 
    crop_size = 224
    models.densenet161(pretrained=True)
    classifier = model.classifier
    input_nodes = 2208

else:
    print("An unsupported model architecture was supplied. Program terminating...")
    exit()


# -------------------- DATA LOADING AND TRANSFORMATIONS --------------------
# Initial transformations for cropping should be dictated by 
# model architecture chosen (resize should always be the same 512)

# Means and stdevs common for pre-trained networks
means = [0.485, 0.456, 0.406]
stdevs = [0.229, 0.224, 0.225]


# -------------------- CLASSIFIER BUILDING --------------------
# Build classifier portion of convolutional neural net to replace
# original ImageNet classifier




# -------------------- CLASSIFIER BUILDING --------------------





# -------------------- START EPOCHS --------------------



    # -------------------- TRAINING --------------------




    # -------------------- VALIDATION --------------------


    


# -------------------- END EPOCHS --------------------





# -------------------- TESTING --------------------



# -------------------- SAVING THE MODEL --------------------



