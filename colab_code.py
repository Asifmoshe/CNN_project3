import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("GPUs:", tf.config.list_physical_devices('GPU'))


from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

zip_path = "/content/drive/MyDrive/content/archive (1).zip"
extract_to = "/content/dataset"

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_to)

print("Extracted to:", extract_to)
print("Top-level:", os.listdir(extract_to)[:20])

import os

def find_dir(root, target_name):
    for r, d, f in os.walk(root):
        if target_name in d:
            return os.path.join(r, target_name)
    return None

train_path = find_dir("/content/dataset", "train")
test_path  = find_dir("/content/dataset", "test")

print("train_path:", train_path)
print("test_path :", test_path)

print("Train folders:", os.listdir(train_path))
print("Test folders :", os.listdir(test_path))


import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

train_dir = "/content/dataset/train"
test_dir = "/content/dataset/test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=25

import json

model_path = "/content/drive/MyDrive/vision_emotion_model.keras"
classes_path = "/content/drive/MyDrive/vision_emotion_classes.json"

model.save(model_path)

with open(classes_path, "w") as f:
    json.dump(class_names, f)

print("Model saved to:", model_path)
print("Classes saved to:", classes_path)