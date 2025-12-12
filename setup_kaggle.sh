#!/bin/bash
# Install Kaggle CLI
pip install kaggle

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ./kaggle.json ]; then
    echo "Error: kaggle.json not found."
    echo "Please place your kaggle.json file in ~/.kaggle/ or in the current directory."
    exit 1
fi

# Move kaggle.json to ~/.kaggle/ if it's in the current directory
if [ -f ./kaggle.json ]; then
    mkdir -p ~/.kaggle
    mv ./kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
fi

# Download the dataset
echo "Downloading dataset..."
kaggle datasets download zalando-research/fashionmnist -p dataset

# Unzip the dataset
echo "Unzipping dataset..."
unzip -o dataset/fashionmnist.zip -d dataset

echo "Dataset setup complete."
