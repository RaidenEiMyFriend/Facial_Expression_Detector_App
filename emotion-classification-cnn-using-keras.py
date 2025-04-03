import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import (Dense, Input, Dropout, GlobalAveragePooling2D, Flatten,
                          Conv2D, BatchNormalization, Activation, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Constants
picture_size = 48
batch_size = 128
data_folder = "data/"
model_folder = "models/"
os.makedirs(model_folder, exist_ok=True)

# Display sample images
expression = 'disgust'
train_path = os.path.join(data_folder, "train", expression)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img_path = os.path.join(train_path, os.listdir(train_path)[i])
    img = load_img(img_path, target_size=(picture_size, picture_size))
    plt.imshow(img)
    plt.axis("off")
plt.show()

# Data Generators
datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()

train_set = datagen_train.flow_from_directory(
    os.path.join(data_folder, "train"),
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_set = datagen_val.flow_from_directory(
    os.path.join(data_folder, "validation"),
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Model Definition
no_of_classes = 7

model = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (5, 5), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),
    
    Dense(512),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),
    
    Dense(no_of_classes, activation='softmax')
])

# Compile Model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Checkpoint to save the best model
checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_folder, "best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train Model
epochs = 50
history = model.fit(
    train_set,
    steps_per_epoch=train_set.n // train_set.batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=test_set.n // test_set.batch_size,
    callbacks=[checkpoint]
)

# Save final model
model.save(os.path.join(model_folder, "final_model.h5"))

# Plot Training Results
plt.style.use('dark_background')
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
