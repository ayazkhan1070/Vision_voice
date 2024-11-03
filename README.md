# Image Captioning with Deep Learning

This repository contains deep learning models for generating captions for images using the Flickr8k dataset. The project combines convolutional and recurrent architectures (DenseNet201 + LSTM) and a Transformer-based model with EfficientNetB0 for image encoding to enhance automated image description capabilities.

## Models

### DenseNet201 + LSTM Model
- **Architecture**: Combines DenseNet201 for feature extraction with an LSTM layer for sequential caption generation.
- **Preprocessing**: Caption normalization, length filtering, and TensorFlow data pipelines were used for efficient training.
- **Performance**: Achieved a BLEU-1 score of 0.20 on the Flickr8k dataset.

### Transformer Model with EfficientNetB0
- **Architecture**: Utilizes EfficientNetB0 for image encoding and a Transformer model with multi-head attention layers for caption generation.
- **Optimization**: Optimized using Sparse Categorical Crossentropy.
- **Performance**: Achieved a BLEU-1 score of 0.51, demonstrating significant improvement over the previous model.

## Dataset
The [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset was used for training and evaluation, providing a rich set of images with human-annotated captions.

## Requirements
- TensorFlow
- Keras
- Numpy
- Matplotlib
- nltk

Install dependencies via:
```bash
pip install -r requirements.txt
