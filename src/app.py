import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageOps
# import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# model = pickle.load('FaceClassifier.h5')

model_path = 'Lukemia Detection.h5'
model = load_model(model_path)

# Function to classify Lukemia
def classify_Lukemia(image):
    # Implement your Lukemia classification logic here
    # Return the result or label
    resize = tf.image.resize(image, (224,224))
    plt.imshow(resize.numpy().astype(int))
    plt.show()
    
    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)
    if (yhat > 0.5).any(): 
        return "Positive"
    else:
        return "Negative"


# Streamlit app
def main():
    st.title("Lukemia Classifer App")
    # st.sidebar.title("Settings")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Perform face classification
        result = classify_Lukemia(image_np)

        # Display the result
        st.write("Predicted class is:", result)

# Run the appbcdc
if __name__ == "__main__":
    main()






