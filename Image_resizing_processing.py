from PIL import Image
import os, sys
path = 'D:/AJ/Aayush/DE/Project/Data/FakeCurrency/2000/'
dirs = os.listdir( path )

def resize():
    i=1
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            _, e = os.path.splitext(path+item)
            f = 'D:/AJ/Aayush/DE/Project/Data/Resized/2000/'
            imResize = im.resize((507,224), Image.ANTIALIAS)
            imResize.save(f + str(i) + e, 'JPEG', quality=90)
            i+=1

resize()