mkdir ~/Downloads/cifar-10-data/SmallTrain
mkdir ~/Downloads/cifar-10-data/SmallTest
mkdir ~/Downloads/cifar-10-data/Train
mkdir ~/Downloads/cifar-10-data/Test

cd ~/Downloads/cifar-10-data/SmallTrain
wget -i ~/Downloads/cifar-10-data/trainSmallPath.txt
cd ~/Downloads/cifar-10-data/SmallTest
wget -i ~/Downloads/cifar-10-data/testSmallPath.txt
cd ~/Downloads/cifar-10-data/Train
wget -i ~/Downloads/cifar-10-data/trainPath.txt
cd ~~/Downloads/cifar-10-data/Test
wget -i ~/Downloads/cifar-10-data/testPath.txt
