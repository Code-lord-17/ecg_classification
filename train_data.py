import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import signal

import os

# Set parameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3
TRAIN_DIR = 'ecg_data'  # Path to your training data directory
SAVE_MODEL_PATH = 'saved_model/stress_analysis.keras'

# Set up ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize the images to [0, 1]
    rotation_range=40,    # Data augmentation: rotation
    width_shift_range=0.2,  # Data augmentation: horizontal shift
    height_shift_range=0.2,  # Data augmentation: vertical shift
    shear_range=0.2,      # Data augmentation: shear
    zoom_range=0.2,       # Data augmentation: zoom
    horizontal_flip=True,  # Data augmentation: flip
    fill_mode='nearest'   # Filling mode for the missing pixels
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',  # For integer labels (Sparse Categorical Crossentropy)
    shuffle=True
)

# Getting the class names
class_names = train_generator.class_indices
class_names = {v: k for k, v in class_names.items()}  # Reverse the dictionary for easier use
print(f'Class Names: {class_names}')

# Build the model
def build_model():
    model = models.Sequential([
        layers.Rescaling(1.0 / 255.0, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')  # Adjust the output layer based on class count
    ])
    return model

# Compile the model
model = build_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1
)

# Save the model
model.save(SAVE_MODEL_PATH)
print(f'Model saved to {SAVE_MODEL_PATH}')
