# Importing Libraries
import AlexNet
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
from glob import glob


# Setting Input
IMAGE_SIZE=[224,224]

train_path='D:/AJ/Aayush/DE/Project/Data/train'
valid_path='D:/AJ/Aayush/DE/Project/Data/test'

# Deep Learning Model
model = AlexNet.built_model()
model.summary()

model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005),
                  metrics=['accuracy'])

# Checkpointing
filepath = 'saved_models/weights-improvement-{epoch : 02d}-{val_acc : .2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

log_csv = CSVLogger('my_logs.csv', separator=' , ', append=False)

callable_list=[checkpoint, log_csv]

# Image Generator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 zoom_range=0.2)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(train_path,
                                               target_size=(256,144),
                                               batch_size=32,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory(valid_path,
                                          target_size=(256,144),
                                          batch_size=32,
                                          class_mode='categorical')


# Model.fit()
r=model.fit_generator(training_set,
                      validation_data=test_set,
                      epochs=5,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set),
                      callbacks=[callable_list])
