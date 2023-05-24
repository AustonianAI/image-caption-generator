\# Image Caption Generator with Streamlit, Hugging Face, and PyTorch

This repository contains a simple yet powerful image captioning application built with Streamlit, Hugging Face's transformers, and PyTorch. The application generates a descriptive caption for any uploaded image.

## Overview

The Image Caption Generator is a demonstration of integrating the Salesforce/blip-image-captioning-large model, a pretrained transformer model from Hugging Face, the PyTorch library, and Streamlit for creating interactive web applications. The application generates captions that provide a textual representation of the images, leading to a more comprehensive and engaging user experience. Importantly, this feature-rich image captioning application is implemented in less than 50 lines of code (excluding whitespace and comments)!

### Key Features

- **Streamlit:** A powerful, fast Python framework used to create the web interface for the application.
- **Hugging Face's Salesforce/blip-image-captioning-large:** A state-of-the-art image captioning AI model that generates the captions for the images.
- **PyTorch:** A Python library used to load and interact with the pretrained AI model.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://image-caption-generator.streamlit.app/)

## How to Run

### Prerequisites

- Python 3.6 or higher
- Streamlit
- PyTorch
- transformers

### Steps

1. Clone this repository: git clone https://github.com/AustonianAI/image-caption-generator.git
2. Navigate to the project directory: cd image-caption-generator
3. Install the necessary Python packages using the command pip install -r requirements.txt.
4. Run the Streamlit app using the command streamlit run app.py.

## Usage

The Image Caption Generator allows users to upload an image. After the image has been uploaded, it generates a descriptive caption for the image based on the content and context of the image.

## Contribution

Contributions, issues, and feature requests are welcome. Feel free to check the Issues page if you want to contribute.

## License

This project is licensed under the terms of the MIT license.
