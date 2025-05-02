import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Must come first before any other Streamlit commands
st.set_page_config(page_title="Cat vs Dog Classifier 🐱🐶", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_dog_classifier_model.h5')  # make sure this path is correct
    return model

model = load_model()

# Page title and description
st.title("Cat vs Dog Classifier")
st.markdown("Upload an image and let the AI tell you whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img_resized = image.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    if prob >= 0.5:
        predicted_class = "Dog 🐶"
        confidence = prob * 100
    else:
        predicted_class = "Cat 🐱"
        confidence = (1 - prob) * 100

    # Display result
    st.subheader("Prediction:")
    st.success(f"{predicted_class}** with *{confidence:.2f}%* confidence")