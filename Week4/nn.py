# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:28:02 2018

@author: enrico
"""

import numpy as np
import matplotlib.pyplot as plt
import load_dataset
import nn_models as nn

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset.load_dataset()

# Example of a picture
#index = 10
#plt.imshow(train_x_orig[index])
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

L_model = True;
##################################TWO LAYERS MODEL############################
### CONSTANTS DEFINING THE MODEL ####
if not L_model:
    print("2 LAYERS MODEL: ")
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    parameters = nn.two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    predictions_train = nn.predict(train_x, train_y, parameters)
    predictions_test = nn.predict(test_x, test_y, parameters)

##################################L LAYERS MODEL############################
### CONSTANTS ###
if L_model:
    print("L LAYERS MODEL: ")
    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    parameters = nn.L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    pred_train = nn.predict(train_x, train_y, parameters)
    pred_test = nn.predict(test_x, test_y, parameters)
    print_mislabeled_images(classes, test_x, test_y, pred_test)
    
