import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource  # cache so it doesn't reload every time
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()

# Class labels (order must match your training set class_indices)
labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# App UI
st.set_page_config(page_title="Brain Tumor Detector 🧠", layout="centered")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image and let the AI predict the tumor type ")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(150,150))  # use (224,224) if you trained with VGG/ResNet
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
