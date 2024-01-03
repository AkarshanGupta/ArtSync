import streamlit as st
from PIL import Image
import os
import torch
import style  # Import your style.py script

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("ArtSync")

img = st.sidebar.selectbox(
    'Select image',
    ('amber.jpg', 'bear.jpeg', 'cat.jpg', 'boat.jpeg', 'ace.jpeg')
)

style_name = st.sidebar.selectbox(
    'Select style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

model_path = os.path.join("saved_models", style_name + ".pth")
input_image_path = os.path.join("images", "content-images", img)
output_image_path = os.path.join("images", "output-images", style_name + "-" + img)

st.write("### Source Image:")
image = Image.open(input_image_path)
st.image(image, width=400)

clicked = st.button("Stylize")

if clicked:
    if os.path.exists(model_path):
        # Load the style model
        style_model = style.load_model(model_path)
        style_model.to(device)
        style_model.eval()

        # Perform style transfer (modify the function name to your implementation)
        style.apply_style_transfer(style_model, input_image_path, output_image_path)

        st.write("### Output Image:")
        output_image = Image.open(output_image_path)
        st.image(output_image, width=400)

        # Add a download button for the output image
        with open(output_image_path, "rb") as f:
            data = f.read()
        st.download_button(
            label="Download Output Image",
            data=data,
            key="download_output_image",
            file_name=os.path.basename(output_image_path),
        )
    else:
        st.write("Model not found. Please check the model path.")
