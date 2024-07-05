import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Predict Pneumonia from Chest X-ray images')
parser.add_argument('--image_path', type=str, help='Path to the X-ray image', required=True)
args = parser.parse_args()

model = load_model("../models/Neumo.h5")

img = cv2.imread(args.image_path)
tempimg = img
img = cv2.resize(img, (300, 300))
img = img / 255.0
img = img.reshape(1, 300, 300, 3)

prediction = model.predict(img) >= 0.5
if prediction:
    label = "Pneumonia"
else:
    label = "Normal"

print("Prediction: " + label)
plt.imshow(tempimg)
plt.title("Prediction: " + label, fontsize=14)
plt.show()
