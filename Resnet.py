import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

import os
import shutil
import json
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image



IMGSIZE=64




TrainData = image_dataset_from_directory('C:/Users/mreda/OneDrive/Desktop/cv classification/Dataset/classification/train',
                                             shuffle=True,
                                              batch_size= 28,
                                             image_size= (IMGSIZE,IMGSIZE),
                                              seed=123
                                             )

total_train_examples = len(TrainData) * 28
print("Total training examples:", total_train_examples)

ValiData = image_dataset_from_directory('C:/Users/mreda/OneDrive/Desktop/cv classification/Dataset/classification/val',
                                             shuffle=True,
                                             batch_size= 28,
                                             image_size= (IMGSIZE,IMGSIZE),
                                             seed=123)



classnames = TrainData.class_names

from tensorflow.keras.layers.experimental import preprocessing

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(preprocessing.RandomFlip('horizontal'))
    data_augmentation.add(preprocessing.RandomRotation(0.2))
    data_augmentation.add(preprocessing.RandomContrast(0.2))

    return data_augmentation

data_augmentation = data_augmenter()

plt.figure(figsize=(10, 10))
for images, labels in TrainData.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classnames[labels[i]])
        plt.axis("off")

model7 = Sequential()


base_pretrainedmodel= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(IMGSIZE,IMGSIZE,3),
                   pooling='avg',classes=5,
                   weights='imagenet')

for layer in base_pretrainedmodel.layers:
        layer.trainable=False

model7.add(base_pretrainedmodel)

from tensorflow.keras.layers import  Flatten , Dense , Dropout
model7.add(Flatten())
model7.add(Dense(512, activation='relu'))
model7.add(Dropout(0.5))
model7.add(Dense(100, activation='softmax'))

model7.summary()


from tensorflow.keras.optimizers import Adam
model7.compile(optimizer=Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])



from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D


inputs = base_pretrainedmodel.input
x = data_augmentation(inputs)

x = base_pretrainedmodel.output


x = Dropout(0.5)(x)
output = Dense(5, activation='softmax')(x)

model4 = Model(inputs=inputs, outputs=output)


base_pretrainedmodel.trainable = True

for layer in base_pretrainedmodel.layers:
  layer.trainable = True


for layer in base_pretrainedmodel.layers[:100]:
    layer.trainable = False

from tensorflow import keras
model7.compile(
    optimizer='rmsprop',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

batch_size = 32


history7 = model7.fit(
    TrainData,
    epochs=10,
    validation_data=ValiData,
)


plt.figure(figsize=(10,3))
plt.plot(history7.history['accuracy'])
plt.plot(history7.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()






