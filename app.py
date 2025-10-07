import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re

# Load the pre-trained model
try:
    model = load_model('my_model.h5')
    # Display the model's expected input shape
    expected_input_shape = model.input_shape[1:]  # Exclude batch dimension
    st.write(f"Expected input shape: {expected_input_shape}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to preprocess and extract features from the URL
def preprocess_url(url):
    """
    Preprocesses the input URL to create feature inputs for the model.
    Features might include the length of the URL, number of digits, special characters, etc.
    """
    features = [
        len(url),  # Length of the URL
        sum(c.isdigit() for c in url),  # Count of digits in URL
        sum(1 for c in url if c in "@_-.")  # Count of special characters in URL
    ]

    # Ensure the features match the model's expected input shape
    # Pad with zeros or trim if necessary
    num_features = expected_input_shape[0]
    if len(features) < num_features:
        features += [0] * (num_features - len(features))  # Pad with zeros
    elif len(features) > num_features:
        features = features[:num_features]  # Trim extra features

    return np.array(features).reshape(1, -1)  # Reshape for model input

# Define the prediction function
def predict_phishing(url):
    features = preprocess_url(url)
    try:
        prediction = model.predict(features)
        # Assuming model outputs a probability for "Phishing"
        return "Phishing" if prediction[0][0] > 0.5 else "Safe"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit app interface
st.title("Web Phishing Detection App")
st.write("Enter a URL below to check if it's safe or a potential phishing attempt.")

# User input
url = st.text_input("Enter URL:")

# Prediction and output
if st.button("Check"):
    if url:
        result = predict_phishing(url)
        if result:
            st.write(f"The URL is likely: **{result}**")
    else:
        st.write("Please enter a valid URL.")