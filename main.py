import streamlit as st

from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# specify the model to be used
hf_model = "Salesforce/blip-image-captioning-large"
# use GPU if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocessor will prepare images for the model
processor = BlipProcessor.from_pretrained(hf_model)

# # then we initialize the model itself
# model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)


def main():
    st.set_page_config(
        page_title='Caption an Image', 
        page_icon=':camera:', 
        layout='wide',
    )

    st.sidebar.header("Caption an Image :camera:")

    uploaded_image = st.sidebar.file_uploader("Upload an image (we aren't saving anything)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        max_width = 500

        width, height = image.size
        if width > max_width:
            ratio = max_width / width
            height = int(height * ratio)
            image = image.resize((max_width, height))

        st.image(image, caption='Uploaded Image.')

        if st.button('Generate Caption'):
            st.write('Caption: A dog is running in the grass.')

if __name__ == '__main__':
    main()