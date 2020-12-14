from PIL import Image
import os, sys

# ["FakeCurrency/500", "FakeCurrency/2000", "RealCurrency/500", "RealCurrency/2000"]

path = 'D:/AJ/Aayush/DE/Project/Data/RealCurrency/2000/'
dirs = os.listdir(path)

def resize():
    i=1
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            _, e = os.path.splitext(path+item)
            f = 'D:/AJ/Aayush/DE/Project/Data/Resized/R2000/'
            imResize = im.resize((256,144), Image.ANTIALIAS)
            imResize.save(f + str(i) + e, 'JPEG', quality=90)
            i+=1

resize()