## Вставить ссылку на картинку в формате jpeg

Пример curl запроса:

```curl -X 'POST' \
    'http://127.0.0.1:8000/predict/' \
    -H 'Content-Type: application/json' \
    -d '{
    "url": "http://images.cocodataset.org/val2017/000000039769.jpg"
}'

На выходе описание изображения
