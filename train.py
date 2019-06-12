# -------------------- IMPORT PACKAGES --------------------

import argparse
import os
from copy import deepcopy
from time import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


# -------------------- SETUP MAJOR INPUT VALUES --------------------

# Setup major inputs required from command line
parser = argparse.ArgumentParser(description='Trains a neural network')
parser.add_argument('data_directory', type=str, 
    help='Filepath for input data of format "data_dir/". \
    Expected to be the parent directory with \
    folders "train", "validation", and "test" inside, with each structured \
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

parser.add_argument('-u', '--hidden_units', nargs='+', type=int,
    default = [512, 256], 
    help = 'Number of nodes to use in each hidden layer, ordered from \
    earliest to latest layer. Not inclusive of the input layer \
    (node count dictated by model architecture chosen) and \
    output layer (always 102 = number of flower labels). \
    Note that usage is --hidden_units count1 count2 count3...')

parser.add_argument('-d', '--dropout', type=bool, 
    default = True, 
    help = 'Determines if dropout with p=0.2 will be used for \
    each hidden layer')

parser.add_argument('-e', '--epochs', type=int, 
    default = 30, 
    help = 'Number of epochs to use for training and validation')

parser.add_argument('-g', '--gpu', type=bool, 
    default = True, 
    help = 'If GPU is available, indicates that it should be used')


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
    print("An unsupported model architecture was supplied. \
        Program terminating...")
    exit()


# Freeze parameters so we don't backprop through the pre-trained 
# feature detector
for param in model.parameters():
    param.requires_grad = False


# -------------------- DATA LOADING AND TRANSFORMATIONS --------------------
# Initial transformations for cropping should be dictated by 
# model architecture chosen (resize should always be the same 512)

# Means and stdevs common for pre-trained networks
means = [0.485, 0.456, 0.406]
stdevs = [0.229, 0.224, 0.225]

data_dir = args.data_directory

# Code here adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

image_transforms = {'train': \
transforms.Compose([transforms.RandomResizedCrop(crop_size),
    transforms.ColorJitter(brightness=0.15, 
        contrast=0.15, 
        saturation=0.15, 
        hue=0),
    transforms.RandomAffine(30),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)]),
'valid': transforms.Compose([transforms.Resize(512),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)]),
'test': transforms.Compose([transforms.Resize(512),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)])
}

phases = ['train', 'valid', 'test']

data = {phase: datasets.ImageFolder(os.path.join(data_dir, phase),
    image_transforms[phase]) for phase in phases}

dataloaders = {phase: torch.utils.data.DataLoader(data[phase], 
    batch_size=64) for phase in phases}

# Set training dataloader to have shuffle = True
dataloaders['train'] = torch.utils.data.DataLoader(data['train'], 
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


if args.arch == 'inception':
    model.fc = classifier
    model_params = model.fc.parameters()
    print(f"Classifier architecture:")
    print(model.fc)

elif args.arch == 'densenet':
    model.classifier = classifier
    model_params = model.classifier.parameters()
    print(f"Classifier architecture:")
    print(model.classifier)



# -------------------- START EPOCHS --------------------

# If GPU is enabled, set device = 'cuda'. Otherwise use CPU
device_to_use = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
device = torch.device(device_to_use)
model.to(device)

# Good loss function to use for LogSoftMax activation layer
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model_params, lr=args.learning_rate)

t0 = time()

# Prep for saving the best epoch's model weights
# Code for this adapted from https://medium.com/datadriveninvestor/creating-a-pytorch-image-classifier-da9db139ba80
from copy import deepcopy

best = {'acc': 0.0, 'epoch': 0, 'weights': deepcopy(model.state_dict())}
epochs = args.epochs

# Used to keep the Udacity online workspace from 
# disconnecting/going to sleep
from workspace_utils import keep_awake

# Keep GPU session awake in Udacity workspace until done training
#epoch_iter = keep_awake(range(epochs))
epoch_iter = range(epochs)


for e in epoch_iter:

    # -------------------- TRAINING --------------------

    # Make sure model is in training mode
    model.train()

    training_loss = 0
    training_batch_counter = 0
    
    for images, labels in dataloaders['train']:

        # Move input and label tensors to the GPU or CPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)

        if args.arch == 'inception': 
            loss = criterion(outputs.logits, labels)
        
        elif args.arch == 'densenet':
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
        # Monitor every 10 batches and final batch
        if training_batch_counter % 10 == 0 or \
        training_batch_counter == (len(dataloaders['train']) - 1):
            print(f"Training batch {training_batch_counter}\nLoss = \
            {training_loss/(training_batch_counter + 1)}\n")
            
        training_batch_counter += 1


    # -------------------- VALIDATION --------------------

    # turn off gradients for speedup in validation
    with torch.no_grad():

        # set model to evaluation mode and remove un-needed things 
        # like Dropout layers
        model.eval()
      
        accuracy = 0
        valid_loss = 0
        val_batch_counter = 0
        
        for images, labels in dataloaders['valid']:
            # Move input and label tensors to the GPU or CPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.exp(outputs)

            _, top_class = probs.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            
            valid_loss += loss.item()
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Monitor every 3 batches and final batch
            if val_batch_counter % 3 == 0 or \
            val_batch_counter == (len(dataloaders['valid']) - 1):
                print(f"Validation batch {val_batch_counter}\nLoss = \
                      {valid_loss/(val_batch_counter + 1)}\n and \
                      accuracy = {accuracy/(val_batch_counter + 1)}\n")
            
            val_batch_counter += 1

    # -------------------- EPOCH REPORTING --------------------

    # Note that normalizing to train/validloader length is due to 
    # need to divide by batch size to effectively average the 
    # quantity in question
    training_loss /= len(dataloaders['train'])
    valid_loss /= len(dataloaders['valid'])
    accuracy /= len(dataloaders['valid'])
    
    print(f"For epoch {e+1}/{epochs}...")
    print(f"{round((time()-t0)/60, 3)} minutes since training started")
    print(f"Training loss = {training_loss}")
    print(f"Validation loss = {valid_loss}")
    print(f"Accuracy = {accuracy}\n\n")
    
    # Update best accuracy and weights if new superior model is found
    if accuracy > best['acc']:
        best['acc'] = accuracy
        best['epoch'] = e+1
        best['weights'] = deepcopy(model.state_dict())
        
        print("Best accuracy updated this epoch \
            to {}\n\n\n".format(best['acc']))

# -------------------- END EPOCHS --------------------

print("Best accuracy found was {} in epoch {}".format(best['acc'],
                                                     best['epoch']))

# Set model weights to the optimal ones found across all epochs
# NOTE: you may get an error 
# IncompatibleKeys(missing_keys=[], unexpected_keys=[])
# This error can be ignored. Model weights were still set properly.
model.load_state_dict(best['weights'])



# -------------------- TESTING --------------------

# turn off gradients for speedup in testing
with torch.no_grad():

    # set model to evaluation mode and remove 
    # un-needed things like Dropout layers
    model.eval()

    test_accuracy = 0
    test_loss = 0

    for images, labels in dataloaders['test']:
        # Move input and label tensors to the GPU or CPU
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.exp(outputs)

        _, top_class = probs.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)

        test_loss += loss.item()
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


# Note that normalizing to train/validloader length is due to need to 
# divide by batch size to effectively average the quantity in question
print(f"Testing loss = {test_loss/len(dataloaders['test'])}")
print(f"Testing accuracy = {test_accuracy/len(dataloaders['test'])}\n\n")


# -------------------- SAVING THE MODEL --------------------

# Note that class_to_idx provides the mapping of my folder names to the 
# index used in the model
if args.arch == 'inception':
    model_arch = models.inception_v3(pretrained=True)

elif args.arch == 'densenet':
    model_arch = models.densenet161(pretrained=True)

checkpoint = {'arch': model_arch,
              'model_state': model.state_dict(),
              'epoch_count': best['epoch'],
              'training_loss': training_loss,
              'validation_loss': valid_loss,
              'opt_state': optimizer.state_dict(),
              'class_to_idx': data['train'].class_to_idx,
              'idx_to_class': {v: k for k,v \
              in data['train'].class_to_idx.items()}}


# Determine the highest number X among the existing checkpoints
# which are assumed to have filenames of the format checkpointX.pth

# Code adapted from 
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

from os import listdir
from os.path import isfile, join

existing_chkpts = [f for f in listdir(args.save_dir) \
                   if isfile(join(args.save_dir, f))]

# Code adapted from 
# https://stackoverflow.com/questions/4666973/how-to-extract-the-substring-between-two-markers

# Take list of existing checkpoint filenames and 
# generate string "checkpointn+1" where n is the highest 
# value used for checkpoint filenames. Guarantees we won't 
# overwrite an existing checkpoint
import re

file_indices = []

for e in existing_chkpts:
    m = re.search('checkpoint(.+).pth', e)
    if m:
        file_indices.append(int(m.group(1)))

# Check that there are any files of proper name scheme in there at all
if file_indices:
    file_n = max(file_indices)
else:
    file_n = 0

save_path = args.save_dir + 'checkpoint' + str(file_n) + '.pth'
torch.save(checkpoint, save_path)


