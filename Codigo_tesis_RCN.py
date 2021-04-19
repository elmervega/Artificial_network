#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:28:16 2021

@author: Aronnvega
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers.convolutional import Conv2D # Import thegroup convolution by Cohen et al.
from keras.layers.normalization import BatchNormalization # ImportBatchNormalisation written to work with group # equi -/ invariant networks
from keras.layers import Conv2D # Import the pooling over the group channels
import numpy as np

def equivariantNetwork(group, padding='valid', large=False):
    # Define the layers that all the networks will use.
    softmax = tf.keras.layers.Activation('softmax')
    maxPooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) 
    globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()

    if not large: # The smaller network structure (3 conv layers)
        # Define the activation layers used after each convolution
        relu1 = tf.keras.layers.Activation('relu') 
        relu2 = tf.keras.layers.Activation('relu') 
        relu3 = tf.keras.layers.Activation('relu')
        
        if group == 'Z2': # This is just an ordinary CNN
            # Define the convolution layers
            conv1 = tf.keras.layers.Conv2D(10, 3, padding=padding) 
            conv2 = tf.keras.layers.Conv2D(10, 3, padding=padding)
            conv3 = tf.keras.layers.Conv2D(10, 3, padding=padding)
            # Define the normalisation and group pooling
            norm = tf.keras.layers.BatchNormalization()
            gPooling = tf.keras.layers.Lambda(lambda x: x) # identity layer since no group action when only Z2
            
        elif group is 'C4': # Network invariant under C4
            # Define the convolutional layers. Note that the number of filters has been reduced to keep the number of
            # parameters roughly the same as for the ordinary CNN
            conv1 = Conv2D(round(10 / 2), 3, 'Z2', group,input_shape=(28, 28, 1), padding=padding)
            conv2 = Conv2D(round(10 / 2), 3, group, group, padding=padding)
            conv3 = Conv2D(10, 3, group, group, padding=padding)
            # Define the normalisation and group pooling
            norm = BatchNormalization(group) 
            gPooling = Conv2D(group)
        else: # Network invariant under D4
            # Define the convolutional layers. Note that the number of filters has been reduced to keep the number of
            # parameters roughly the same as for the ordinary CNN
            conv1 = Conv2D(round(10 / 3), 3, 'Z2', group, input_shape=(28, 28, 1), padding=padding)
            conv2 = Conv2D(round(10 / 3), 3, group, group, padding=padding)
            conv3 = Conv2D(10, 3, group, group, padding=padding)
            # Define the normalisation and group pooling
            norm = BatchNormalization(group) 
            gPooling = Conv2D(group)
            
        # Use Keras functional API to construct the network with the layers defined above
        inputs = keras.Input(shape=(28, 28, 1)) 
        
        x = conv1(inputs)
        x = relu1(x)
        x = conv2(x)
        x = relu2(x)
        x = maxPooling(x)
        x = norm(x, training=False) 
        x = conv3(x)
        x = relu3(x)
        x = gPooling(x)
        x = globalAvgPooling(x) 
        outputs = softmax(x)
        
    else: # The larger network structure with 7 convolutional layers
        # Define the activation layers used after each convolution
        relu1 = tf.keras.layers.Activation('relu') 
        relu2 = tf.keras.layers.Activation('relu') 
        relu3 = tf.keras.layers.Activation('relu')
        relu4 = tf.keras.layers.Activation('relu')
        relu5 = tf.keras.layers.Activation('relu')
        relu6 = tf.keras.layers.Activation('relu')
        relu7 = tf.keras.layers.Activation('relu')
        
        if group == 'Z2': # This is just an ordinary CNN
        
            # Define the convolution layers
            conv1 = tf.keras.layers.Conv2D(10, 3, padding=padding)
            conv2 = tf.keras.layers.Conv2D(10, 3, padding=padding) 
            conv3 = tf.keras.layers.Conv2D(10, 3, padding=padding) 
            conv4 = tf.keras.layers.Conv2D(10, 3, padding=padding) 
            conv5 = tf.keras.layers.Conv2D(10, 3, padding=padding) 
            conv6 = tf.keras.layers.Conv2D(10, 3, padding=padding)
            conv7 = tf.keras.layers.Conv2D(10, 3, padding=padding)
            # Define the normalisation and group pooling
            norm = tf.keras.layers.BatchNormalization()
            gPooling = tf.keras.layers.Lambda(lambda x: x) # identity layer since no group action when only Z2
            
        elif group is 'C4': # Network invariant under C4
            # Define the convolutional layers. Note that the number of filters has been reduced to keep the number of
            # parameters roughly the same as for the ordinary CNN
            conv1 = Conv2D(round(10 / 2), 3, 'Z2', group, input_shape=(28, 28, 1), padding=padding)
            conv2 = Conv2D(round(10 / 2), 3, group, group, padding=padding)
            conv3 = Conv2D(round(10 / 2), 3, group, group, padding=padding)
            conv4 = Conv2D(round(10 / 2), 3, group, group, padding=padding)
            conv5 = Conv2D(round(10 / 2), 3, group, group, padding=padding) 
            conv6 = Conv2D(round(10 / 2), 3, group, group, padding=padding)
            conv7 = Conv2D(10, 3, group, group, padding=padding)
            # Define the normalisation and group pooling
            norm = BatchNormalization(group) 
            gPooling = Conv2D(group)
            
        else: # Network invariant under D4
            # Define the convolutional layers. Note that the number of filters has been reduced to keep the number of
            # parameters roughly the same as for the ordinary CNN
            conv1 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, 'Z2', group, input_shape=(28, 28, 1), padding=padding)
            conv2 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, group, group, padding=padding)
            conv3 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, group, group, padding=padding)
            conv4 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, group, group, padding=padding)
            conv5 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, group, group, padding=padding)
            conv6 = Conv2D(int(np.floor(10 / np.sqrt(8))), 3, group, group, padding=padding)
            conv7 = Conv2D(10, 3, group, group, padding=padding)
            # Define the normalisation and group pooling
            norm = BatchNormalization(group)
            gPooling = Conv2D(group)
            
        # Use Keras functional API to construct the network with the layers defined above
        inputs = keras.Input(shape=(28, 28, 1))
        x = conv1(inputs) 
        x = relu1(x)
        x = conv2(x)
        x = relu2(x)
        x = maxPooling(x)
        x = norm(x, training=False) 
        x = conv3(x)
        x = relu3(x)
        x = conv4(x)
        x = relu4(x) 
        x = conv5(x)
        x = relu5(x)
        x = conv6(x)
        x = relu6(x)
        x = conv7(x)
        x = relu7(x)
        x = gPooling(x)
        x = globalAvgPooling(x) 
        outputs = softmax(x)
        # Finally actually create the model
        model = keras.Model(inputs=inputs , outputs=outputs)
        return model # Return the constructed model
    
    
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
# Importing the MNIST data set
mnist = keras.datasets.mnist
(train_images , train_labels), (test_images , test_labels) = mnist.load_data()

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1) 

train_images = train_images /255.0
test_images = test_images /255.0
# Define the network to be used
group = 'C4'
model = equivariantNetwork(group=group) 
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy']) # Final set up of model

model.fit(train_images[:],train_labels[:],epochs=5,validation_data=(test_images ,test_labels)) # Train the model

test_loss , test_acc = model.evaluate(test_images ,test_labels) # Evaluate after training



# Test of invariance (due to the rotation invariant global average pool) by manually rotating a single image
tmp = test_images[0:4,:,:,:] 
tmp[1]=np.rot90(tmp[0],k=1) 
tmp[2]=np.rot90(tmp[0],k=2) 
tmp[3]=np.rot90(tmp[0],k=3)

images = np.array(model(tmp).numpy()) # Feed the rotated images through the model
# Plot the result from the model for the differently rotated images , if network is invariant these should be the same

input_shape = np.shape(images)
numChan = 1
numImgs = len(images)
fig, axes = plt.subplots(nrows=numImgs, ncols=numChan)
for ax, ind in zip(axes.flatten(), range(numChan * numImgs)): 
    image = images[ind, :]
    ax.plot(image)
    ax.set_ylabel('Image nr. ' + str(ind))


plt.show(block=True)