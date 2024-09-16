import streamlit as st
import matplotlib.pyplot as plt
import torch
from colorizers import *
import skimage.color as color
from PIL import Image
import numpy as np

# Streamlit app setup
st.title("Image Colorization App")

use_gpu = False

# Load colorizers

def load_colorizers():
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    return colorizer_eccv16, colorizer_siggraph17

colorizer_eccv16, colorizer_siggraph17 = load_colorizers()

# Move models to GPU if needed
if use_gpu and torch.cuda.is_available():
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Convert RGBA to RGB if needed
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Display the original image
    st.image(img, caption="Original Image", use_column_width=True)

    # Grab L channel in both original and resized resolutions (256x256)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if use_gpu and torch.cuda.is_available():
        tens_l_rs = tens_l_rs.cuda()

    # Convert RGB image to LAB
    img_lab = color.rgb2lab(img)

    # Colorizer outputs 256x256 ab map; resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Display the results
    st.write("Colorization Results")

    st.image([img_bw, out_img_eccv16, out_img_siggraph17],
             caption=["Input (Grayscale)", "Output (ECCV 16)", "Output (SIGGRAPH 17)"],
             use_column_width=True)
    
    # Option to download colorized images
    out_img_eccv16_pil = Image.fromarray((out_img_eccv16 * 255).astype(np.uint8))
    out_img_siggraph17_pil = Image.fromarray((out_img_siggraph17 * 255).astype(np.uint8))


else:
    st.write("Please upload an image to colorize.")
