target = open("/Users/xiaodong/Downloads/target", "rw+")
result = open("/Users/xiaodong/Downloads/result", "rw+")
Xtest = open("/Users/xiaodong/Downloads/cifar10data/X_test.txt", "rw+")

targetList = [int(x)-1 for x in target.readlines()]
resultList = [int(x)-1 for x in result.readlines()]
Xtest = [int(x) for x in Xtest.readlines()]

dictionary = dict(zip(targetList, resultList))

f = open("niubiRES", "w")
for i in xrange(10000):
    f.write(str(dictionary[Xtest[i]]) + "\n")
f.close()