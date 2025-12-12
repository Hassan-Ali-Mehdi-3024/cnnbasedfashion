import io
import pathlib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="👕", layout="centered")

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

MODEL_PATH = pathlib.Path("models/fashion_mnist_cnn.h5")

@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please train/export the model first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image (or any image—we will convert and resize) to get a prediction.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert to grayscale, resize to 28x28, normalize to [0,1]
    image = image.convert("L").resize((28, 28))
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

if uploaded:
    bytes_data = uploaded.read()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption="Uploaded image", use_column_width=True)

    input_tensor = preprocess_image(image)
    preds = model.predict(input_tensor)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2%}")
else:
    st.info("Upload an image to see the prediction.")
