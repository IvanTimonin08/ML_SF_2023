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
[![Tests](https://github.com/IvanTimonin08/ML_SF_2023/blob/main/.github/workflows/python-app.yml/badge.svg)](https://github.com/IvanTimonin08/ML_SF_2023/blob/main/.github/workflows/python-app.yml)
# Vision Transformer (base-sized model) 
http://127.0.0.1:8000/predict_image_class?image_url=http://images.cocodataset.org/val2017/000000039769.jpg
Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224. It was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. and first released in [this repository](https://github.com/google-research/vision_transformer). However, the weights were converted from the [timm repository](https://github.com/rwightman/pytorch-image-models) by Ross Wightman, who already converted the weights from JAX to PyTorch. Credits go to him. 

Disclaimer: The team releasing ViT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The Vision Transformer (ViT) - это модель кодировщика трансформера (похожая на BERT), предварительно обученная на большой коллекции изображений в надзорной манере, а именно на ImageNet-21k, с разрешением 224x224 пикселя. Затем модель была дообучена на ImageNet (также известном как ILSVRC2012), наборе данных, включающем 1 миллион изображений и 1000 классов, также с разрешением 224x224.

Путем предварительного обучения модель изучает внутреннее представление изображений, которое затем можно использовать для извлечения признаков, полезных для последующих задач: если у вас, например, есть набор данных с размеченными изображениями, вы можете обучить стандартный классификатор, разместив линейный слой поверх предварительно обученного кодировщика. Обычно линейный слой размещают поверх токена [CLS], так как последнее скрытое состояние этого токена можно рассматривать как представление всего изображения.

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=google/vit) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/model_doc/vit.html#).

## Training data

The ViT model was pretrained on [ImageNet-21k](http://www.image-net.org/), a dataset consisting of 14 million images and 21k classes, and fine-tuned on [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/), a dataset consisting of 1 million images and 1k classes. 

## Training procedure

### Preprocessing

The exact details of preprocessing of images during training/validation can be found [here](https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py). 

Images are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

### Pretraining

The model was trained on TPUv3 hardware (8 cores). All model variants are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet, the authors found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224.

## Evaluation results

For evaluation results on several image classification benchmarks, we refer to tables 2 and 5 of the original paper. Note that for fine-tuning, the best results are obtained with a higher resolution (384x384). Of course, increasing the model size will result in better performance.

### BibTeX entry and citation info

```bibtex
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
```
