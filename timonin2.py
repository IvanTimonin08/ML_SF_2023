import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import io

st.title("Image Classification with ViT")

# Create a file uploader to allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Read the uploaded file and open it as an image
    image = Image.open(uploaded_file)

    # Load the processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Process the image and make predictions
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Display the uploaded image and the predicted class
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicted class:", predicted_class)
else:
    st.write("Please upload an image.")
