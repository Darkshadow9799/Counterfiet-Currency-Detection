from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

PATH = 'D:/AJ/Aayush/DE/Project/Data/FakeCurrency/500/1.jpg' # add path here
NEW_PATH = 'D:/AJ/Aayush/DE/Project/Data/FakeCurrency/Fake_500/' # saving dir
PREFIX = 'f500' # prefix of image

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2,
    brightness_range=[0.5,1.5]
)

img = load_img(PATH)
x = img_to_array(img)
x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=NEW_PATH, save_prefix=PREFIX, save_format='jpeg'):
    i+=1
    if i > 20:
        break