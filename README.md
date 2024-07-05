# Pneumonia Detection using Convolutional Neural Network

## Project Description :

This project uses Convolutional Neural Networks (CNN) to detect Pneumonia from chest X-ray images. The dataset is sourced from the Kaggle dataset "Chest X-Ray Images (Pneumonia)" by Paul Mooney.

## Dataset :

The dataset consists of X-ray images categorized into two classes:
  -> **Normal**
  -> **Pneumonia**

You can download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Usage : 

<> Create a virtual environment and activate it.

    python3 -m venv venv
    source venv/bin/activate

<> Download the dataset and place it in the data/ directory and run the ipy notebook file.

## Model Architecture :
  The CNN model consists of several convolutional layers followed by max-pooling layers, and dense layers for classification. The model is trained using the Adam optimizer and binary cross-entropy loss function.

## Results :
<> The model achieved:

Training accuracy : 98.16%

Validation accuracy : 100%

