'''
Processes images for use in inference via an artificial neural network
'''

import numpy as np
from PIL import Image

def calculate_resize(image, goal_size):
    '''
    Takes a PIL image and outputs the new dimensions it should have in a PyTorch-like
    manner such that the shorter axis is set to goal_size and the long axis is scaled 
    such that the original aspect ratio is maintained. This is intended to be fed into
    Image.resize().
    
    Parameters
    ----------
    image: PIL image object
    
    goal_size: int. Desired minimum dimension size.
    
    
    Returns
    -------
    tuple of ints of the format (new width, new height)
    '''
    
    aspect_ratio = min(image.size) / max(image.size)

    # Get the index of the smaller dimension
    min_ix = image.size.index(min(image.size))

    # width is smaller than height
    if min_ix == 0:
        return (goal_size, int(goal_size/aspect_ratio))
    else:
        return (int(goal_size/aspect_ratio), goal_size)




def calculate_centercrop_box(image, goal_size):
    '''
    Takes a PIL image and outputs the 4-tuple of coordinates needed to perform a centered cropping
    of the image of size goal_size x goal_size. This is intended to be fed into Image.crop().
    
    Parameters
    ----------
    image: PIL image object
    
    goal_size: int. Desired resultant dimension size.
    
    
    Returns
    -------
    4-tuple of ints coordinates of the format (new_width_X, new_width_Y 
                                                new_height_X, new_height_Y)
    '''
    
    width = image.size[0]
    height = image.size[1]
    
    # upperLeft_X, upperLeft_Y, lowerRight_X, lowerRight_Y
    bound_box = [int(0.5*(width - goal_size)), int(0.5*(height - goal_size)),
                int(0.5*(width + goal_size)), int(0.5*(height + goal_size))]
    
    # Check to make sure both dimensions produce goal_size
    bound_box_w = bound_box[2] - bound_box[0]
    bound_box_h = bound_box[3] - bound_box[1]
    
    if bound_box_w != goal_size:
        bound_box[2] -= bound_box_w - goal_size
    
    
    return tuple(bound_box)



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor object
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Note that, in order to make this similar to my validation and testing batches,
    # I actually need to resize to 512x512 and then crop to 299x299 pixels
    
    # Resize to 512 on the shortest axis, keeping aspect ratio
    new_size = calculate_resize(image, 512)
    image = image.resize(new_size)   
    
    
    # Center-crop the image to 299x299
    crop_size = calculate_centercrop_box(image, 299)
    image = image.crop(crop_size)
    
    
    # Convert image to numpy array
    np_image = np.array(image)
    
    # Scale 0-255 color channel values to 0-1
    np_image = np.divide(np_image, np.array([255]))
    
    # Normalize color channel values
    MEANS = np.array([0.485, 0.456, 0.406])
    STDEVS = np.array([0.229, 0.224, 0.225])
    
    np_image = np.subtract(np_image, MEANS)
    np_image = np.divide(np_image, STDEVS)
    
    # Move color dimension from third to first place, retaining other two dimensions' orders
    # Resulting array should be of shape (3,299,299)
    np_image = np_image.transpose(2,0,1)
    
    # Return numpy array
    return torch.Tensor(np_image)