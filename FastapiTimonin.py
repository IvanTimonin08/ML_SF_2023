from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

# Загрузка процессора и модели ViT
processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.post("/predict_image_class")
async def predict_image_class(image: UploadFile = File(...)):
    try:
        # Чтение и обработка изображения
        contents = await image.read()
        img = Image.open(BytesIO(contents))

        # Обработка изображения и выполнение предсказаний
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # Модель предсказывает один из 1000 классов ImageNet
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}
