import streamlit as st
import src.archs.unet_arch as unet_arch
import src.archs.deeplab_arch as deeplab_arch
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
import numpy as np
from main import generate_mask_img, get_image_from_array

st.title("Semantic Segmentation of Flood Images")

# Select the model architecture
model_arch = st.selectbox("Select the model", ["Unet", "DeepLabV3"])

# Load the model
if model_arch == "Unet":
    model = unet_arch.UNET(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load("src/models/deeplab-01.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    
elif model_arch == "DeepLabV3":
    model = deeplab_arch.DeepLabV3().to(device)
    checkpoint = torch.load("src/models/unet-01.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    
else:
    model = unet_arch.UNET(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load("src/models/unet-01.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)

if file is not None:
    with col1:
        image = Image.open(file)
        image = image.resize((480, 320))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
gen_button = st.button("Generate Mask", use_container_width=True)

if gen_button:
    with col2:
        image = np.array(Image.open(file))
        print(image)
        image = get_image_from_array(img = image)
        mask = generate_mask_img(img = image, model = model, device = device) 
        st.image(mask, caption="Generated Mask", use_column_width=True)