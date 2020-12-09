from PIL import Image
import os, sys

# ["FakeCurrency/500", "FakeCurrency/2000", "RealCurrency/500", "RealCurrency/2000"]

path = 'D:/AJ/Aayush/DE/Project/Data/RealCurrency/500/'
dirs = os.listdir(path)

def resize():
    i=1
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            _, e = os.path.splitext(path+item)
            f = 'D:/AJ/Aayush/DE/Project/Data/Resized/R500/'
            imResize = im.resize((507,224), Image.ANTIALIAS)
            imResize.save(f + str(i) + e, 'JPEG', quality=90)
            i+=1

resize()