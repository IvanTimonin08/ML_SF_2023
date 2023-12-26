from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests
from fastapi import FastAPI
from PIL import Image
from io import BytesIO

app = FastAPI()

# Загрузка процессора и модели ViT
processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.get("/predict_image_class")
async def predict_image_class(image_url: str):
    try:
        # Загрузка изображения по предоставленному URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Обработка изображения и выполнение предсказаний
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # Модель предсказывает один из 1000 классов ImageNet
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}
