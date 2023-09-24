import os
import sys
import re
import torch
import torch.onnx
import torchvision.transforms as transforms
from PIL import Image
from transformer_net import TransformerNet  # You need to import your TransformerNet class
import utils  # You need to import your utility functions

# Define a function for style transfer
def apply_style_transfer(model, content_image_path, output_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the content image
    content_image = Image.open(content_image_path).convert('RGB')

    # Preprocess the content image
    content_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    # Perform style transfer
    with torch.no_grad():
        output = model(content_image).cpu()

    # Save the stylized output image
    utils.save_image(output_image_path, output[0])

# Define a function to check and create directories
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not os.path.exists(args.checkpoint_model_dir):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

# Define a function to load a trained style transfer model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the TransformerNet model
    style_model = TransformerNet()

    # Load the model's state dictionary from the provided path
    state_dict = torch.load(model_path, map_location=device)

    # Remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]

    # Load the state dictionary into the model
    style_model.load_state_dict(state_dict)

    # Move the model to the appropriate device (CPU or GPU)
    style_model.to(device)

    # Set the model in evaluation mode
    style_model.eval()

    return style_model

