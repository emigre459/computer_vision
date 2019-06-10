# -------------------- SETUP MAJOR INPUT VALUES --------------------

import argparse

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
    default = 'inception_v3', 
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

# -------------------- DATA LOADING AND TRANSFORMATIONS --------------------
# Initial transformations for cropping should be dictated by 
# model architecture chosen (resize should always be the same 512)


