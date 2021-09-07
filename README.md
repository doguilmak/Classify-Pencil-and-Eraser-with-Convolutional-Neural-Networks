
# Pencil and Eraser Classification using Convolutional Neural Network

## Problem Statement

The task at hand is to **classify erasers and pencils** using supervised machine learning methods (CNNs). Output layer is binary.

## Dataset

The dataset was created by me. A total of 30 pictures, including 15 eraser and 15 pencil photographs, were used as tests and training.

***You can add different kind of eraser and pencil photos to prevent overfit or you can change type of the classification object.***

## Methodology

For understanding the methodology you are free to visit the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website. 

## Analysis

| Layer (type)    | Output Shape |  Param # |
|--|--|--|
| conv2d_3 (Conv2D) | (None, 62, 62, 32) | 896
| max_pooling2d_3 (MaxPooling2 | (None, 31, 31, 32)    | 0
| conv2d_4 (Conv2D) | (None, 29, 29, 32)  | 9248
| max_pooling2d_4 (MaxPooling2) | (None, 14, 14, 32)  | 0
| flatten_2 (Flatten)  | (None, 6272)  | 0
| dense_3 (Dense)   | (None, 128) | 802944
| dense_4 (Dense)   | (None, 1)   | 129 

> Total params: 813,217
> Trainable params: 813,217
> Non-trainable params: 0
> Took 90.84954714775085 seconds to classificate objects.

## How to Run Code

Before running the code if you don't have all the necessery libraries, you need to have these libraries:

 - keras 
 - numpy 
 - pandas 
 - warnings 
 - sklearn
    
## Contact With Me

If you have something to say to me please contact me on [Twitter](https://twitter.com/Doguilmak). 
