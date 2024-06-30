--># Brain MRI Image Classification for Brain Tumor Detection

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images for brain tumor detection.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Running the Project](#running-the-project)
5. [Results](#results)
6. [License](#license)

## Overview

This project involves:
- Downloading and preparing the dataset.
- Preprocessing the images.
- Building and training a CNN using the VGG16 architecture.
- Evaluating the model's performance.

## Requirements

- Python 3.6+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- TQDM
- Kaggle API

## Setup and Installation

### Step 1: Install Dependencies

First, install the necessary Python packages:

```bash
pip install tensorflow keras opencv-python numpy matplotlib tqdm kaggle
```

### Step 2: Setup Kaggle API

1. Download your `kaggle.json` API key file from the Kaggle website.
2. Upload the `kaggle.json` file to your working directory.

### Step 3: Download the Dataset

Run the following commands in your Colab notebook or terminal to download the dataset:

```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
```

### Step 4: Extract the Dataset

```python
from zipfile import ZipFile

with ZipFile('brain-mri-images-for-brain-tumor-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/brain_mri_images')
```

## Running the Project

### Step 1: Preprocess the Images

```python
import os
import cv2
from tqdm.notebook import tqdm
import numpy as np

X = []
y = []

# Load 'yes' images
os.chdir('/content/brain_mri_images/yes')
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append(1)  # Label 1 for 'yes'

# Load 'no' images
os.chdir('/content/brain_mri_images/no')
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append(0)  # Label 0 for 'no'

X = np.array(X)
y = np.array(y)
```

### Step 2: Split the Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

### Step 3: Encode the Labels

```python
from sklearn import preprocessing
import tensorflow as tf

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
```

### Step 4: Build the Model

```python
from keras.applications import vgg16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense

vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg.layers:
    layer.trainable = False

def build_model(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

num_classes = 2
FC_Head = build_model(vgg, num_classes)
model = Model(inputs=vgg.input, outputs=FC_Head)
```

### Step 5: Compile and Train the Model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=5,
                    validation_data=(X_test, y_test),
                    verbose=1)
```

## Results

### Step 1: Plot Training and Validation Accuracy

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
