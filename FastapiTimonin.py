from fastapi import FastAPI, UploadFile, File
from PIL import Image
import requests
import io
from transformers import ViTImageProcessor, ViTForImageClassification

app = FastAPI()

# Загрузка процессора и модели ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.post("/predict_image_class")
async def predict_image_class(image: UploadFile = File(...)):
    try:
        # Чтение и обработка изображения
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Обработка изображения и выполнение предсказаний
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}
