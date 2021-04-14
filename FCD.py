# Importing Libraries

from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger,ModelCheckpoint
from glob import glob


# Setting Input

IMAGE_SIZE=[256,144]

train_path='/content/drive/MyDrive/Project/Data/train'
valid_path='/content/drive/MyDrive/Project/Data/test'


# Deep Learning Model
model=MobileNetV2(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

for layers in model.layers:
  layers.trainable=False

folders=glob('/content/drive/MyDrive/Project/Data/train/*')

x=Flatten()(model.output)

prediction=Dense(len(folders),activation='softmax')(x)

model=Model(inputs=model.input,outputs=prediction)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# Checkpointing

filepath = 'saved_models/weights-improvement-{epoch:02d}.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

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

history=model.fit_generator(training_set,
                      validation_data=test_set,
                      epochs=50,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set),
                      callbacks=callable_list)
