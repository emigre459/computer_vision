import argparse
import torch

parser = argparse.ArgumentParser(description="Reports back important metadata \
    saved in a checkpoint.pth file")

# Setup major inputs required from command line
parser.add_argument('input_filename', type=str, 
    help='Filepath for input image for which the flower \
    name will be predicted')

# Setup optional parameters that can be entered from the command line
parser.add_argument('-s', '--save_dir', type=str, 
    default = 'model_checkpoints/', 
    help = 'Filepath indicating where trained model checkpoint files \
    are stored')

args = parser.parse_args()


# If GPU is enabled, set device = 'cuda'. Otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = torch.device("cpu")

filepath = args.save_dir + args.input_filename
checkpoint = torch.load(filepath, map_location=device)

# Can be Inception3 or DenseNet based on options in train.py
model_name = checkpoint['arch'].__class__.__name__
epoch = checkpoint['epoch_count']
val_acc = checkpoint['validation_accuracy']
test_acc = checkpoint['test_accuracy']

print(f"Checkpoint: {args.input_filename}")
print(f"Model: {model_name}")
print(f"Best validation accuracy of {val_acc} found at epoch {epoch}")
print(f"Test accuracy of model is {test_acc}")

print(f"Fields in checkpoint include:\n\n{checkpoint.keys()}")