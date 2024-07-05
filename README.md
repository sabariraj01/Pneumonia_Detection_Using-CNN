# Pneumonia Detection using Convolutional Neural Network

## Project Description :

This project uses Convolutional Neural Networks (CNN) to detect Pneumonia from chest X-ray images. The dataset is sourced from the Kaggle dataset "Chest X-Ray Images (Pneumonia)" by Paul Mooney.

## Dataset :

The dataset consists of X-ray images categorized into two classes:
  * **Normal**
  * **Pneumonia**

You can download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Usage : 

<> Create a virtual environment and activate it.

    python3 -m venv venv
    source venv/bin/activate

<>Install the required packages.

    pip install -r requirements.txt

<> Download the dataset and place it in the data/ directory and run the ipy notebook file.

## Model Architecture :

  The CNN model architecture consists of several convolutional layers, each followed by max-pooling layers, and finally dense layers for classification.
  
<> Summary of the model :

* Conv2D layers with ReLU activation
* MaxPooling2D layers
* Flatten layer
* Dense layers with ReLU activation
* Output Dense layer with sigmoid activation for binary classification

_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 298, 298, 16)      448       
max_pooling2d (MaxPooling2D (None, 149, 149, 16)      0         
conv2d_1 (Conv2D)           (None, 147, 147, 32)      4640      
max_pooling2d_1 (MaxPooling (None, 73, 73, 32)        0         
conv2d_2 (Conv2D)           (None, 71, 71, 64)        18496     
max_pooling2d_2 (MaxPooling (None, 35, 35, 64)        0         
conv2d_3 (Conv2D)           (None, 33, 33, 128)       73856     
max_pooling2d_3 (MaxPooling (None, 16, 16, 128)       0         
conv2d_4 (Conv2D)           (None, 14, 14, 128)       147584    
max_pooling2d_4 (MaxPooling (None, 7, 7, 128)         0         
flatten (Flatten)           (None, 6272)              0         
dense (Dense)               (None, 256)               1605888   
dense_1 (Dense)             (None, 512)               131584    
dense_2 (Dense)             (None, 1)                 513       
=================================================================
Total params: 1983009 (7.56 MB)
Trainable params: 1983009 (7.56 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________


## Results :
<> The model achieved:

* Training accuracy : 98.16%
* Validation accuracy : 100%

