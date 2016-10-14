import numpy as np
import re

if __name__ == '__main__':
    def clean(Str):
        newStr = re.sub("\[\"\[\'", "", Str)
        newStr = re.sub("\'\"\, \'", ",", newStr)
        newStr = re.sub("[\]\\\'\]]", "", newStr)
        newStr = re.sub("n", "", newStr)
        newStr = re.sub("\\\\", "", newStr)

        return newStr

    test = open("/Users/songyang/Documents/CSCI8360/Project3/TestWithIdLabel3.txt")
    train = open("/Users/songyang/Documents/CSCI8360/Project3/TrainReadyNum.txt")

    f1 = open("/Users/songyang/Documents/CSCI8360/Project3/trainPixelData", "w")
    for line in train:
        f1.write(clean(line))
    f1.close()

    f2 = open("/Users/songyang/Documents/CSCI8360/Project3/testPixelData3", "w")
    for line in test:
        f2.write(clean(line))
    f2.close()

