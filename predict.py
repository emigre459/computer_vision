'''
Infers the name of a flower from an image supplied to it
by utilizing a pre-trained artificial neural network
'''


# -------------------- IMPORT PACKAGES --------------------
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from process_image import process_image
from PIL import Image

# -------------------- SETUP MAJOR INPUT VALUES --------------------

parser = argparse.ArgumentParser(description="Predicts a flower's name, \
    given its image")

# Setup major inputs required from command line
parser.add_argument('input_filepath', type=str, 
    help='Filepath for input image for which the flower \
    name will be predicted')

parser.add_argument('checkpoint_filepath', type=str, 
    help='Filepath of the checkpoint.pth file to be used for \
    loading up a trained model to use for inference')


# Setup optional parameters that can be entered from the command line
parser.add_argument('-t', '--top_k', type=int, metavar='',
    default = 1, 
    help = 'Number of classes to return as inferences of the flower name. \
    Returned names are ordered from most probable/certain to least')

parser.add_argument('-g', '--gpu', type=bool, metavar='',
    default = True, 
    help = 'If GPU is available, indicates that it should be used')

parser.add_argument('-c', '--category_names', type=str, metavar='',
    default=None,
    help = 'Filepath of JSON file that defines the mapping from\
     folder name (usually an integer as str) to actual flower name')

parser.add_argument('-d', '--display', type=bool, metavar='',
    default = False, 
    help = 'If True, displays input image with its ground truth \
    flower name as well as the top_k classes predicted as the \
    flower name, with the latter displayed as a bar chart')


args = parser.parse_args()


# -------------------- LOAD MODEL FROM CHECKPOINT --------------------

# If GPU is enabled, set device = 'cuda'. Otherwise use CPU
if torch.cuda.is_available() and args.gpu:
    device = torch.device("cuda:0")

elif args.gpu and not torch.cuda.is_available():
    print("GPU unavailable, using CPU\n")
    device = torch.device("cpu")

else:
    device = torch.device("cpu")

checkpoint = torch.load(args.checkpoint_filepath, map_location=device)

model = checkpoint['arch']
# Freeze parameters so we can't backprop through the 
# pre-trained feature detector
for param in model.parameters():
    param.requires_grad = False

# Can be Inception3 or DenseNet based on options in train.py
model_name = model.__class__.__name__

# Use model name to dictate how you setup classifier
if model_name == 'Inception3':
    model.fc = checkpoint['classifier']

elif model_name == 'DenseNet':
    model.classifier = checkpoint['classifier']

# Load weights, biases, and other attributes
model.load_state_dict(checkpoint['model_state'])
model.class_to_idx = checkpoint['class_to_idx']
model.idx_to_class = checkpoint['idx_to_class']

model.to(device)
model.eval()


# -------------------- PREDICTION --------------------

# Load the Image
image = Image.open(args.input_filepath)

# Preprocess the image
processed_image = process_image(image)


# PyTorch expects a 4D tensor for input into Conv2d layers, with a size-1
# first dimension. Not entirely sure what it is, but probably the batch ID.
# So let's just give it a new dimension with nothing in it and hope for the best!
data = processed_image.unsqueeze(0).to(device)

# Feedforward the image input to generate prediction probabilities
with torch.no_grad():
    
    outputs = model(data)
    probs = torch.exp(outputs)

top_ps, top_idx = probs.topk(args.top_k, dim = 1)

# Convert to lists for iterating
top_ps = list(top_ps.numpy()[0])
top_idx = list(top_idx.numpy()[0])

# Translate model label indices to image folder labels
top_classes = [model.idx_to_class[idx] for idx in top_idx]

# top classes are folder labels, unless a JSON mapping is passed
# to category_names
if args.category_names:
    import json

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    top_classes = [cat_to_name[name] for name in top_classes]


print(f"Top classes predicted, in order, are {top_classes}\
 with probabilities {top_ps}")

# Display flower image with its actual name as title and 
# top_k predicted classes with bar chart showing their 
# probabilities
if args.display:

    from imshow import imshow
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # NOTE: much of the plotting code is adapted from 
    # https://medium.com/@rayheberer/generating-matplotlib-subplots-programmatically-cc234629b648
    
    # Setup the multi-object Figure
    fig, (ax1, ax2) = plt.subplots(figsize = (15,6), ncols=1, nrows=2)
    fig.suptitle('What flower is that, anyways?')
    
    img_path = args.input_filepath

    # Setup the processed flower image + label title
    # Use class folder name as flower name unless category_names is passed
    flower_name = cat_to_name[img_path.split('/')[-2]] if \
    args.category_names else img_path.split('/')[-2]

    ax1.set(title=flower_name)

    #TODO: will it be a problem that processed image is unsqueezed?
    imshow(processed_image, ax = ax1)
    ax1.axis('off')
    

    # Setup the probability bar chart
    sns.barplot(x = top_ps, y = top_classes, orient = 'h', ax = ax2, 
        color = 'limegreen')

    ax2.invert_yaxis()
    ax2.set(title='Educated Guesses', xlabel='Probability')

    #print()
    plt.show()