import streamlit as st
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("mnist_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]

# ---------------------------
# Neural Network Functions
# ---------------------------
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return A2

def make_prediction(image_array):
    A2 = forward_prop(W1, b1, W2, b2, image_array)
    return np.argmax(A2), np.max(A2)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 grayscale)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Normalize
    img_array = img_array / 255.0
    img_array = img_array.reshape(784, 1)

    prediction, confidence = make_prediction(img_array)

    st.success(f"🎯 Prediction: {prediction}")
    st.info(f"Confidence: {confidence*100:.2f}%")