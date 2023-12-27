## Вставить ссылку на картинку в формате jpeg

Пример curl запроса:

```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_image_class' \
  -H 'accept: application/json' \
  -F 'image=@/path/to/your/image.jpg'
```

На выходе описание изображения
