from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from tqdm import tqdm

query = ["Fake_500", "Fake_2000", "Real_500","Real_2000"]

NEW_PATH = 'D:/AJ/Aayush/DE/Project/Data/'+query[2]+'/'
PREFIX = 'R500' # prefix of image

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.5,1.5]
)

query2 = ["F500","F2000","R500","R2000"]

for i in tqdm(range(1, 22)):
    PATH = 'D:/AJ/Aayush/DE/Project/Data/Resized/'+query2[2]+'/'+str(i)+'.jpg'
    img = load_img(PATH)
    x = img_to_array(img)
    x = x.reshape((1, ) + x.shape)

    i = 1
    for batch in datagen.flow(x, batch_size=1, save_to_dir=NEW_PATH, save_prefix=PREFIX, save_format='jpg'):
        i+=1
        if i > 20:
            break