# Fashion MNIST CNN Classifier - AI Agent Instructions

## Project Overview
A complete ML pipeline for Fashion MNIST classification featuring:
- **Training notebook**: [cnnbasedfashion.ipynb](cnnbasedfashion.ipynb) - Full training, evaluation, and visualization workflow
- **Streamlit web app**: [app.py](app.py) - Production inference interface with ~99% accuracy
- **Model files**: `models/` directory containing `.keras` and legacy `.h5` formats

The project classifies 10 fashion item categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Environment Setup

### System Requirements
- **Python**: 3.11+ (installed via deadsnakes PPA on Ubuntu)
- **GPU**: NVIDIA RTX 6000 Ada (CUDA 12.x compatible)
- **OS**: Ubuntu 22.04 LTS

### Quick Setup (New System)
```bash
# Install Python 3.11 (Ubuntu)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Create and activate virtual environment
cd /root/cnnbasedfashion
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Activating the Environment
```bash
cd /root/cnnbasedfashion && source venv/bin/activate
```

## Architecture & Key Design Patterns

### Model Loading Strategy (Critical)
The app uses **multi-strategy fallback loading** to handle Keras version compatibility issues. When modifying [app.py](app.py):

1. **Primary**: Load `.h5` format (most compatible across Keras 2.x/3.x)
2. **Fallback 1**: Rebuild architecture in `rebuild_model_architecture()`, then load weights
3. **Fallback 2**: Try Keras 3.x `safe_mode=False`
4. **Fallback 3**: Load with custom objects for old activation names

**Why**: Models saved in older Keras versions often fail to load in newer versions due to serialization changes.

**Example from [app.py](app.py#L40-L100)**:
```python
# Rebuild architecture matching notebook's model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    # ... rest of architecture
])
```

### Data Preprocessing Convention
**Critical invariant**: All images must be preprocessed to match training format:
- Convert to grayscale (`.convert("L")`)
- Resize to 28×28 using `Image.Resampling.LANCZOS`
- Normalize to [0,1]: `array.astype("float32") / 255.0`
- Reshape to `(1, 28, 28, 1)` for batch dimension

See `preprocess_image()` in [app.py](app.py#L166-L175) - this function is **critical** for inference accuracy.

### Class Names Order
The class index mapping is **hardcoded** and must match across all files:
```python
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
This order is defined in both [app.py](app.py#L16) and [cnnbasedfashion.ipynb](cnnbasedfashion.ipynb). **Never reorder** without retraining.

## Development Workflows

### Running the Web Application
```bash
streamlit run app.py
```
The app auto-opens at `http://localhost:8501` with:
- Wide layout mode for side-by-side image comparison
- Cached model loading with `@st.cache_resource`
- Real-time confidence visualization using pandas DataFrames

### Training a New Model
Execute [cnnbasedfashion.ipynb](cnnbasedfashion.ipynb) sequentially:
1. **Data acquisition**: Auto-loads from `dataset/` CSV files or downloads via TensorFlow Datasets
2. **Model training**: CNN with 2 Conv2D layers (32→64 filters) + MaxPooling
3. **Evaluation**: Generates confusion matrix, classification report, per-class accuracy
4. **Export**: Saves to `models/fashion_mnist_cnn.keras` AND `.h5` for compatibility

**Important**: Always export in **both** `.keras` and `.h5` formats for maximum compatibility.

### Setting Up Kaggle Data
If dataset not present locally:
```bash
bash setup_kaggle.sh  # Requires kaggle.json in ~/.kaggle/ or current directory
```
Downloads to `dataset/` and auto-unzips. The notebook's `load_data()` function checks this path first.

## Project-Specific Conventions

### Dependency Versions (Updated Dec 2024)
[requirements.txt](requirements.txt) uses latest compatible versions:
- `tensorflow[and-cuda]==2.18.0` - Latest TensorFlow with bundled CUDA 12.x support
- `keras==3.13.0` - Modern Keras 3.x API (bundled with TF 2.18)
- `numpy==2.0.2` - NumPy 2.x for performance
- `streamlit==1.41.1` - Latest Streamlit
- `scikit-learn==1.6.0` - For evaluation metrics

**GPU Support**: TensorFlow 2.18+ bundles CUDA libraries via `tensorflow[and-cuda]`, no manual CUDA installation needed.

### Model File Duplication
Both `fashion_mnist_cnn.keras` and `fashion_mnist_cnn.h5` exist in `models/`:
- `.keras`: Modern format (TF 2.x native)
- `.h5`: Legacy format for backwards compatibility

**Pattern**: Save in both formats after training to avoid deployment issues.

### Streamlit UI State Management
The app is **stateless** - each upload triggers full re-inference. No session state persistence.
- File uploader resets on re-run
- Model loads once via `@st.cache_resource`
- Predictions computed on-demand with `verbose=0` to suppress TensorFlow output

## Critical Integration Points

### Model ↔ Notebook Contract
The notebook's model architecture (lines 169-177 in [cnnbasedfashion.ipynb](cnnbasedfashion.ipynb)) **must match** `rebuild_model_architecture()` in [app.py](app.py#L24-L38):
- Same layer sequence
- Same filter counts (32, 64)
- Same activation functions
- Same input shape (28, 28, 1)

**Break this contract → inference fails**

### Streamlit ↔ TensorFlow Integration
- Use `st.spinner()` during `model.predict()` for UX feedback
- Set `verbose=0` on predict to avoid console spam in Streamlit
- Display preprocessed 28×28 image upscaled to 280×280 with `NEAREST` resampling for pixel visibility

## Testing & Validation

### Verifying Model Output
Expected behavior:
- Input shape: `(1, 28, 28, 1)`
- Output shape: `(1, 10)` - softmax probabilities
- Output sum: ≈ 1.0 (probability distribution)
- Predicted class: `np.argmax(preds[0])`

### Common Issues & Fixes

**Issue**: "Model file not found"
**Fix**: Run notebook training cells to generate models, or check `MODEL_PATH` in [app.py](app.py#L18)

**Issue**: Model loading fails with serialization errors
**Fix**: The app's fallback strategy handles this automatically. If all strategies fail, retrain with current TF version

**Issue**: Low confidence predictions
**Fix**: Image likely differs from Fashion MNIST distribution (complex background, color, multiple items). App displays warning if confidence < 0.7

## File Reference Map
- **[app.py](app.py)**: Production inference app (Streamlit)
- **[cnnbasedfashion.ipynb](cnnbasedfashion.ipynb)**: Training, evaluation, visualization
- **[requirements.txt](requirements.txt)**: Pinned dependencies
- **[setup_kaggle.sh](setup_kaggle.sh)**: Dataset download automation
- **`models/`**: Saved model files (.keras, .h5)
- **`dataset/`**: Fashion MNIST CSV files (not in repo, download via script)
