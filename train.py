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
    model = models.inception_v3(pretrained=True)
    #classifier = model.fc
    input_nodes = 2048


elif args.arch == 'densenet': 
    crop_size = 224
    model = models.densenet161(pretrained=True)
    #classifier = model.classifier
    input_nodes = 2208

else:
    print("An unsupported model architecture was supplied. Program terminating...")
    exit()


# Freeze parameters so we don't backprop through the pre-trained feature detector
for param in model.parameters():
    param.requires_grad = False


# -------------------- DATA LOADING AND TRANSFORMATIONS --------------------
# Initial transformations for cropping should be dictated by 
# model architecture chosen (resize should always be the same 512)

# Means and stdevs common for pre-trained networks
means = [0.485, 0.456, 0.406]
stdevs = [0.229, 0.224, 0.225]

data_dir = 'data/'

# Code here adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

image_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(crop_size),
    transforms.ColorJitter(brightness=0.15, 
        contrast=0.15, 
        saturation=0.15, 
        hue=0),
    transforms.RandomAffine(30),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)]),
'validation': transforms.Compose([transforms.Resize(512),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)]),
'test': transforms.Compose([transforms.Resize(512),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)])
}

phases = ['train', 'validation', 'test']

data = {phase: datasets.ImageFolder(os.path.join(data_dir, phase),
    image_transforms[phase]) for phase in phases}

dataloaders = {phase: torch.utils.data.DataLoader(data[phase], 
    batch_size=64) for phase in phases}

# Set training dataloader to have shuffle = True
dataloaders['train'] = torch.utils.data.DataLoader(data[phase], 
    batch_size=64, shuffle = True)


# -------------------- CLASSIFIER BUILDING --------------------
# Build classifier portion of convolutional neural net to replace
# original ImageNet classifier
classifier = nn.Sequential()
nodes = args.hidden_units

classifier.add_module('hidden1', nn.Linear(input_nodes, nodes[0]))

for i, _ in enumerate(nodes):
    if i+1 >= len(nodes): break

    classifier.add_module('activation' + str(i+1), nn.ReLU())
    if args.dropout: classifier.add_module('dropout' + str(i+1), 
        nn.Dropout(0.2))
    classifier.add_module('hidden' + str(i+2), nn.Linear(nodes[i], nodes[i+1]))

classifier.add_module('activation' + str(i+1), nn.ReLU())
classifier.add_module('output', nn.Linear(nodes[-1], 102))
classifier.add_module('activation_output', nn.LogSoftmax(dim=1))


if arch == 'inception':
    model.fc = classifier

elif arch == 'densenet':
    model.classifier = classifier


# -------------------- START EPOCHS --------------------



    # -------------------- TRAINING --------------------




    # -------------------- VALIDATION --------------------





# -------------------- END EPOCHS --------------------





# -------------------- TESTING --------------------



# -------------------- SAVING THE MODEL --------------------



