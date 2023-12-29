---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
- imagenet-21k
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---
[![Tests](https://github.com/IvanTimonin08/ML_SF_2023/actions/workflows/python-app.yml/badge.svg)](https://github.com/IvanTimonin08/ML_SF_2023/actions/workflows/python-app.yml)
# Модель Vision Transformer (ViT) (модель базового размера)

ViT - это модель кодировщика трансформера, предварительно обученная на большой коллекции изображений в надзорной манере. Она была предварительно обучена на наборе данных ImageNet-21k, который содержит 14 миллионов изображений и 21 843 класса, с разрешением 224x224 пикселя. Затем модель была дообучена на ImageNet 2012 (также известном как ILSVRC2012), который включает в себя 1 миллион изображений и 1000 классов, также с разрешением 224x224.

Путем предварительного обучения модель изучает внутреннее представление изображений, которое затем можно использовать для извлечения признаков, полезных для последующих задач классификации изображений.

## Назначение и ограничения

Вы можете использовать эту модель для классификации изображений. Вы также можете найти в модельном хабе модифицированные версии[model hub](https://huggingface.co/models?search=google/vit), которые подходят для конкретных задач, которые вас интересуют.



### Как использовать

Для классификации изображения из набора данных COCO 2017 на один из 1000 классов ImageNet вы можете использовать следующий код:

```python
import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import io

st.title("Image Classification with ViT")

# Создание загрузчика файлов, чтобы пользователь мог загрузить изображение
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Проверка, было ли загружено изображение
if uploaded_file is not None:
    # Чтение загруженного файла и открытие его как изображения
    image = Image.open(uploaded_file)

    # Загрузка обработчика и модели
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Обработка изображения и выполнение предсказаний
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Отображение загруженного изображения и предсказанного класса
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.write("Предсказанный класс:", predicted_class)
else:
    st.write("Пожалуйста, загрузите изображение.")
```

Для получения дополнительных примеров кода обратитесь к документации. [documentation](https://huggingface.co/transformers/model_doc/vit.html#).

## Данные об обучении

Модель ViT была предварительно обучена на наборе данных ImageNet-21k [ImageNet-21k](http://www.image-net.org/), который содержит 14 миллионов изображений и 21k классов, а также дообучена на ImageNet [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/), который содержит 1 миллион изображений и 1k классов.

## Процедура обучения

- **Предобработка:** Изображения изменяются/пересчитываются до одного разрешения (224x224) и нормализуются по каналам RGB со средним (0.5, 0.5, 0.5) и стандартным отклонением (0.5, 0.5, 0.5).

- **Предварительное обучение:** Модель была обучена на аппаратном обеспечении TPUv3 (8 ядер). Все варианты модели обучаются с размером пакета 4096 и скоростью обучения с разогревом в 10 тыс. шагов. Для ImageNet авторы нашли полезным также применять обрезку градиента при глобальной норме 1. Разрешение обучения - 224.

- **Результаты оценки:** Для результатов оценки на нескольких бенчмарках классификации изображений обратитесь к таблицам 2 и 5 оригинальной статьи. Обратите внимание, что для дообучения лучшие результаты достигаются при более высоком разрешении (384x384). Увеличение размера модели также приведет к лучшей производительности.

Подробности предварительной обработки изображений во время обучения/валидации можно найти [here](https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py). 




