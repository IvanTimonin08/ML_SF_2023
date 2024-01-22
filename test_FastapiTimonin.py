import io
from PIL import Image
import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification

def test_image_classification_app():
    # Создание тестового изображения
    test_image = Image.new('RGB', (300, 300))

    # Загрузка обработчика и модели
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Обработка изображения и выполнение предсказаний
    inputs = processor(images=test_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Проверка, что предсказанный класс соответствует ожидаемому результату
    assert predicted_class == "expected_class", "Ошибка: предсказанный класс не соответствует ожидаемому результату"

