import streamlit as st
from PIL import Image
import os
import torch
import style  # Import your style.py script

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Pytorch Style Transfer")

# Allow the user to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

style_name = st.sidebar.selectbox(
    'Select style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

model_path = os.path.join("saved_models", style_name + ".pth")

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Check if the model exists
    if os.path.exists(model_path):
        # Load the style model
        style_model = style.load_model(model_path)
        style_model.to(device)
        style_model.eval()

        # Perform style transfer
        stylized_image = style.apply_style_transfer(style_model, uploaded_image)

        # Display the stylized image
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)
    else:
        st.write("Model not found. Please check the model path.")


