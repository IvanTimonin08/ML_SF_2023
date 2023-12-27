import pytest
import requests
from fastapi.testclient import TestClient
from FastapiTimonin import app

client = TestClient(app)

def test_predict_image_class():
    # Создание фиктивного файла изображения для теста
    image_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06...'
    files = {'image': ('test_image.jpg', image_data, 'image/jpeg')}

    # Отправка POST-запроса к эндпоинту /predict_image_class с использованием requests
    response = requests.post("http://testserver/predict_image_class", files=files)

    # Проверка кода ответа
    assert response.status_code == 200

    # Проверка содержимого ответа
    data = response.json()
    assert "predicted_class" in data
