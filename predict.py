# -------------------- IMPORT PACKAGES --------------------
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


# -------------------- SETUP MAJOR INPUT VALUES --------------------

parser = argparse.ArgumentParser(description="Predicts a flower's name, \
    given its image")

# Setup major inputs required from command line
parser.add_argument('input_filepath', type=str, 
    help='Filepath for input image for which the flower \
    name will be predicted')

parser.add_argument('checkpoint', type=str, 
    help='Filepath of the checkpoint.pth file to be used for \
    loading up a trained model to use for inference')


# Setup optional parameters that can be entered from the command line
parser.add_argument('-t', '--top_k', type=int, 
    default = 1, 
    help = 'Number of classes to return as inferences of the flower name. \
    Returned names are ordered from most probable/certain to least')

parser.add_argument('-c', '--category_names', type=str, 
    default = 'cat_to_name.json', 
    help = 'Filepath of JSON file that defines the mapping from\
     folder name (usually an integer) to actual flower name')

parser.add_argument('-g', '--gpu', type=bool, 
    default = True, 
    help = 'If GPU is available, indicates that it should be used')


