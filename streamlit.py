import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# Load the image from the URL
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Load the processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Process the image and make predictions
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Display the predicted class using Streamlit
st.image(image, caption='Input Image', use_column_width=True)
st.write("Predicted class:", predicted_class)
