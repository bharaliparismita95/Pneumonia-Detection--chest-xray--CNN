from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as Backend
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data specifications
image_width, image_height = 150, 150
train_data = 'E:/Pneumonia Detection/Chest-xray data/train_images/'
valid_data = 'E:/Pneumonia Detection/Chest-xray data/valid_images/'
test_data = 'E:/Pneumonia Detection/Chest-xray data/test_images/'

batch_size = 128

if Backend.image_data_format() == 'channels_first':
    input_shape = (3, image_width, image_height)
else:
    input_shape = (image_width, image_height, 3)

# creating the model
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(train_data,
                                                  target_size=(image_width, image_height),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  class_mode='binary')
validation_data = valid_datagen.flow_from_directory(valid_data,
                                                    target_size=(image_width, image_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
testing_data = test_datagen.flow_from_directory(test_data,
                                                target_size=(image_width, image_height),
                                                batch_size=1,
                                                class_mode=None,
                                                shuffle=False)

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=2, mode='min')
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, mode='max')
epochs = 20

# training the model
history = model.fit(training_data,
                    steps_per_epoch=len(training_data),
                    epochs=epochs,
                    validation_data=validation_data,
                    validation_steps=len(validation_data),
                    callbacks=[earlystop, lr_reduce])

# saving the model
model.save('cnn_model.h5')

# plotting the training and validation accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('model_accuracy.jpg')
plt.show()

# plotting the training and validation loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('model_loss.jpg')
plt.show()

