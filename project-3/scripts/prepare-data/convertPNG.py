#python script for converting 32x32 pngs
from PIL import Image
import os
from array import *

#data = array('d')
data = list()

# -----------------------
# mapping label to number
# -----------------------
idFile = open("/Users/xiaodong/Downloads/metadata/X_train.txt", "rw+")
idList = [int(x) for x in idFile.readlines()]
idFile.close()

labelFile = open("/Users/xiaodong/Downloads/y_train.txt", "rw+")
labelList = [int(x) for x in labelFile.readlines()]
labelFile.close()

idLableDict = dict(zip(idList, labelList))

path = '/Users/xiaodong/Downloads/cifar10data/Train'
for dirname, dirnames, filenames in os.walk(path):
    for filename in sorted(filenames):
        if filename.endswith('.png'):

            #grab the image
            im = Image.open(os.path.join(dirname, filename))
            pix = im.convert("RGB")

            class_name = idLableDict[int(filename.split(".")[0])]
            data.append(class_name)
            #Extract RGB from pixels and append
            #note: first we get red channel, then green then blue
            for color in range(0,3):
                for x in range(0,32):
                    for y in range(0,32):
                        data.append(pix.getpixel((x,y))[color])


#write all to text
output_file = open('/Users/xiaodong/Downloads/cifar10data/TrainReady.txt', 'w+')
for i in range(10000):
    output_file.write(str(data[(i*3073):((i+1)*3073)])+"\n")
output_file.close()






