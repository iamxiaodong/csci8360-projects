# csci8360-p3-tooYoungTooSimple

#  CIFAR-10 Image Classification with Deep Residual Networks on Torch

This is project 3 in CSCI 8360 course at University of Georgia, Spring 2016. In this project we were challenged to predict the class labels of the test set from CIFAR-10 image dataset with 10 different labeled categories. The dataset consist of 60,000 PNG images with 50,000 in the training set and 10,000 in the test set. We conducted a number of manipulations on the data and applied the Deep Residual Networks with 200 layers by Kaiming He to train the classification model. In the end, the model achieves 92.82% prediction accuracy in the test set.

## Getting Started

Here are the instructions to run the .lua scripts with torch on AWS EC2 GPU instance. Be sure to change the data path when you run the scripts locally.


### Installing
**[Install CUDA 7.5 and cuDNN v5](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#optional-install-cuda-gpus-on-linux)**

**[Install Torch on Ubuntu 14.04 Amazon EC2 GPU Instances with CUDA GPU](https://github.com/brotchie/torch-ubuntu-gpu-ec2-install)**

### Configuration
**[Getting started with Torch](http://torch.ch/docs/getting-started.html)**


## Residual Neural Network
Here is the reference paper:

**[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)**

The basic building block of Deep Residual Learning network is:

![residualunit](https://github.com/eds-uga/csci8360-p3-tooYoungTooSimple/blob/master/residualunit.png?raw=true)

###Model Training
* Convert the PNGs to trainDataPixel and testDataPixel. (convertPNG.py and cleanPNGVectors.py) 
* Convert the pixel data to the RNN input format. (cifar10-gen.lua)
* Transform the data by normalization, centralize, crop and so on. (transforms.lua)
* Train the data by train.lua which use resnet-nn-pre.lua as specified model.
* 
```
th main.lua -netType residual-nn-pre -depth 119 -batchSize 64 -nGPU 1 -nThreads 4 -dataset cifar10 -nEpochs 250 -shareGradInput false
```
* The depth of network should be (9n+2). However, the depth shouldn't be too deep since there might be a memory leak.

###Prediction

* Modify the 'Trainer:test()' function in train.lua, add output part to output targets(filenames) and results(labels 1-10). Since the input is shuffled in loading part, we must output filenames as well.
* 
```
th main.lua -netType residual-nn-pre -depth 119 -batchSize 64 -nGPU 1 -nThreads 4 -dataset cifar10 -nEpochs 250 -shareGradInput false -testOnly true -resume ~/torchTest/checkpoints
```
* Reorder the result by 'X_test.txt', 'target' and 'result'. (changeResultOrder.py)


## Running
To run the code, be sure change the data path and install all dependent lua modules. Also you need to keep scripts and folders in folder 'RNN_scripts' together. Now run the main.lua as follow:

```
th main.lua -netType residual-nn-pre -depth 119 -batchSize 64 -nGPU 1 -nThreads 4 -dataset cifar10 -nEpochs 250 -shareGradInput false 
```

Test the data by existing checkpoints:

```
th main.lua -netType residual-nn-pre -depth 119 -batchSize 64 -nGPU 1 -nThreads 4 -dataset cifar10 -nEpochs 250 -shareGradInput false -testOnly true -resume ~/torchTest/checkpoints
```

For the meaning of options please see 'opts.lua' file.

## Authors

* **[Xiaodong Jiang](https://www.linkedin.com/in/xiaodongjiang)** - Ph.D. Student, *Department of Statistics*
* **[Yang Song](https://www.linkedin.com/in/yang-song-74298a118/en)** - M.S. Student, *Department of Statistics*
* **[Yaotong Cai](https://www.linkedin.com/in/yaotong-colin-cai-410ab026)** - Ph.D. Candidate, *Department of Statistics*

## Acknowledgments

* Thanks all team members for the laborious work and great collaboration.
* Thanks [Dr. Quinn](http://cobweb.cs.uga.edu/~squinn/).
