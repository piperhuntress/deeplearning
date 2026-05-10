import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

#Load the dataset
(images_train,labels_train), (images_test,labels_test )=tf.keras.datasets.mnist.load_data()
# Display dataset shapes
print("Training Images:", images_train.shape)
print("Testing Images:", images_test.shape)
print("testing labels:", labels_test.shape)
images_train = images_train/255.0
images_test = images_train/255.0

# Display first 5 training images
plt.figure(figsize=(10,2)) # 10-width, 2=height in inches
size=len(images_train)


# plt.subplot(1, 2, 1)
# #display 1st image
# plt.imshow(images_train[0])

# #last image
# plt.subplot(1, 2, 2)
# plt.imshow(images_train[size-1])
# plt.show()

#Create a subplot

#using a loop

for i in range(5):
  plt.subplot(1,5,i+1)
  plt.imshow(images_train[i])

plt.show()

# Create a basic NN
model =models.Sequential()
model.add(layers.Input())
