import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2

st.set_page_config(page_title="Brain Tumor Detection 🧠", layout="centered")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image and get the tumor prediction instantly 💕")

# Load model safely
@st.cache_resource  # caches the model so it doesn't reload every time
def load_brain_tumor_model():
    try:
        # Try loading the .keras format first (preferred)
        model = tf.keras.models.load_model("brain_tumor_model.keras")
    except:
        # Fallback to .h5 with MobileNetV2 custom object
        model = tf.keras.models.load_model(
            "brain_tumor_model.h5",
            custom_objects={"MobileNetV2": MobileNetV2}
        )
    return model

model = load_brain_tumor_model()

# Class labels (must match training order)
labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(150,150))  # match training size
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
