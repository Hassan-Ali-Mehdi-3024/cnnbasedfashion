# Fashion MNIST Classification (End Semester Project)

## Overview
A Convolutional Neural Network (CNN) to classify Fashion-MNIST images (10 classes). Includes local CSV loading (Kaggle download) with a TFDS fallback.

## Tech Stack
- Python 3.10+
- TensorFlow 2.15
- TensorFlow Datasets 4.9
- NumPy, Pandas, Matplotlib
- Streamlit (deployment)
- Pillow (image preprocessing)
- Kaggle CLI (for dataset download)

## Project Structure
- `fashion-mnist-with-cnn-accuracy-99.ipynb` — main notebook (data load, preprocessing, model, training, evaluation, predictions)
- `app.py` — Streamlit app for inference using the exported model
- `dataset/` — expected location for downloaded CSV/IDX files (created by Kaggle download)
- `setup_kaggle.sh` — helper script to download/unzip the dataset via Kaggle CLI
- `requirements.txt` — pinned dependencies

## Setup
1) Create/activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Configure Kaggle credentials (if using Kaggle download):
- Place `kaggle.json` in `~/.kaggle/` (chmod 600) or in the repo root before running the setup script.

4) Download dataset via Kaggle (optional if you already have `dataset/` populated):
```bash
./setup_kaggle.sh
```
This will populate `dataset/` with `fashion-mnist_train.csv`, `fashion-mnist_test.csv`, and IDX files.

## Running the Notebook
- Open `fashion-mnist-with-cnn-accuracy-99.ipynb` in VS Code / Jupyter.
- Run cells top-to-bottom. The notebook will load from `dataset/` if present; otherwise it falls back to `tfds.load('fashion_mnist')`.

## Key Steps in the Notebook
- Data load: CSV (Kaggle) or TFDS fallback
- Preprocessing: normalization, caching
- Visualization: sample images and labels
- Model: Sequential CNN (Conv2D/MaxPool/Flatten/Dense)
- Training: 20 epochs (configurable), batching/shuffling
- Evaluation: test accuracy/loss
- Prediction & analysis: sample predictions with plots

## Exporting the Model
- Run the training cells in the notebook.
- Run the "Model Export" cell to save `models/fashion_mnist_cnn.h5`.

## Running the Streamlit App
1) Ensure `models/fashion_mnist_cnn.h5` exists (export from notebook as above).
2) Install deps: `pip install -r requirements.txt`
3) Launch: `streamlit run app.py`
4) Upload an image (any format); the app converts to 28x28 grayscale and returns the predicted class and confidence.

## Notes
- CPU execution works out of the box; GPU requires proper CUDA/cuDNN setup.
- If you see tqdm/ipywidgets warnings, install `ipywidgets` in your environment to silence them.
