import io
import pathlib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd

st.set_page_config(
    page_title="Fashion MNIST Classifier", 
    page_icon="👕", 
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

MODEL_PATH = pathlib.Path("models/fashion_mnist_cnn.keras")

@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please train/export the model first.")
        st.stop()
    
    # Load model with compatibility for old Keras format
    # Custom objects to handle old activation function names and ignore batch_shape
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'softmax_v2': tf.nn.softmax,
                'relu': tf.nn.relu
            },
            compile=False  # Don't compile to avoid compatibility issues
        )
        
        # Recompile with current Keras settings
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Trying alternative loading method...")
        
        # Fallback: Try loading with safe_mode=False for older formats
        try:
            import h5py
            # Try loading the .h5 version if it exists
            h5_path = pathlib.Path("models/fashion_mnist_cnn.h5")
            if h5_path.exists():
                model = tf.keras.models.load_model(h5_path, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )
                return model
        except:
            pass
        
        st.error("Could not load model. Please check the model file format.")
        st.stop()

model = load_model()

# Sidebar with model information
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    ### Fashion MNIST CNN Classifier
    
    **Dataset:** Fashion MNIST  
    **Classes:** 10 fashion categories  
    **Input:** 28×28 grayscale images  
    **Architecture:** Convolutional Neural Network  
    **Accuracy:** ~99%
    
    ---
    
    ### 📋 Supported Classes
    """)
    
    for i, class_name in enumerate(CLASS_NAMES, 1):
        st.markdown(f"{i}. {class_name}")
    
    st.markdown("""
    ---
    
    ### 🎯 Tips for Best Results
    
    - Use images with **simple backgrounds**
    - Ensure the **item is centered**
    - **Grayscale or color** images work
    - Any size accepted (auto-resized to 28×28)
    - Works best with items similar to training data
    
    ---
    
    ### ⚙️ Automatic Processing
    
    Your uploaded image is automatically:
    1. ✅ Converted to grayscale
    2. ✅ Resized to 28×28 pixels
    3. ✅ Normalized to [0, 1] range
    4. ✅ Reshaped to (1, 28, 28, 1)
    """)

# Main content
st.title("👕 Fashion MNIST Classifier")
st.markdown("""
This AI-powered application classifies fashion items into 10 categories using a deep learning CNN model 
trained on the Fashion MNIST dataset with ~99% accuracy.
""")

st.markdown("---")

# File uploader
uploaded = st.file_uploader(
    "📤 Upload an image of a fashion item",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    help="Upload any image - it will be automatically preprocessed to match model requirements"
)

def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    Preprocess image for model prediction.
    Returns: (preprocessed_array, preprocessed_image_for_display)
    """
    # Convert to grayscale and resize to 28x28
    processed_img = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array and normalize to [0,1]
    arr = np.array(processed_img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    
    return arr, processed_img

if uploaded:
    bytes_data = uploaded.read()
    image = Image.open(io.BytesIO(bytes_data))
    
    # Display original and preprocessed images side by side
    st.subheader("🖼️ Image Processing")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(image, caption=f"Original size: {image.size[0]}×{image.size[1]} pixels", use_container_width=True)
        st.info(f"**Format:** {image.format or 'Unknown'} | **Mode:** {image.mode}")
    
    with col2:
        st.markdown("**Preprocessed for Model**")
        input_tensor, processed_display = preprocess_image(image)
        # Scale up the 28x28 for better visibility
        display_img = processed_display.resize((280, 280), Image.Resampling.NEAREST)
        st.image(display_img, caption="Grayscale 28×28 (shown at 280×280)", use_container_width=True)
        st.success("✅ Converted to grayscale, resized to 28×28, normalized to [0,1]")
    
    st.markdown("---")
    
    # Make prediction
    with st.spinner("🔮 Classifying..."):
        preds = model.predict(input_tensor, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
    
    # Display results
    st.subheader("🎯 Classification Results")
    
    # Main prediction with large display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h2 style='color: #1f77b4; margin: 0;'>{CLASS_NAMES[pred_idx]}</h2>
            <h3 style='color: #666; margin: 10px 0 0 0;'>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Detailed probability distribution
    st.subheader("📊 Confidence Scores for All Classes")
    
    # Create DataFrame for all predictions
    pred_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Confidence': preds[0] * 100
    }).sort_values('Confidence', ascending=False)
    
    # Display bar chart
    st.bar_chart(pred_df.set_index('Class')['Confidence'])
    
    # Display detailed table
    with st.expander("📈 View Detailed Probabilities"):
        pred_df['Confidence'] = pred_df['Confidence'].apply(lambda x: f"{x:.2f}%")
        pred_df['Rank'] = range(1, len(pred_df) + 1)
        st.dataframe(
            pred_df[['Rank', 'Class', 'Confidence']].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    
    # Interpretation help
    if confidence > 0.9:
        st.success("🎉 High confidence prediction! The model is very certain about this classification.")
    elif confidence > 0.7:
        st.info("👍 Good confidence. The model is fairly certain about this classification.")
    else:
        st.warning("⚠️ Low confidence. The image may be ambiguous or unlike training data.")
        st.markdown("**Suggestion:** Try a clearer image with better contrast and centered subject.")
    
else:
    st.info("👆 Upload an image to get started with classification!")
    
    # Sample instructions when no file uploaded
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Click the **Browse files** button above
    2. Select any image containing a fashion item
    3. The app will automatically process and classify it
    4. View the results with confidence scores
    
    **Note:** The model works best with images similar to Fashion MNIST dataset - 
    simple grayscale images of individual clothing items or accessories on plain backgrounds.
    """)
