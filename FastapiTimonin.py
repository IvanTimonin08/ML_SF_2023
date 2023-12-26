from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/predict_image_class")
async def predict_image_class(image_url: 'http://images.cocodataset.org/val2017/000000039769.jpg'):
    # Загрузка изображения по предоставленному URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Загрузка процессора и модели ViT
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Обработка изображения и выполнение предсказаний
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # Модель предсказывает один из 1000 классов ImageNet
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return {"predicted_class": predicted_class}
