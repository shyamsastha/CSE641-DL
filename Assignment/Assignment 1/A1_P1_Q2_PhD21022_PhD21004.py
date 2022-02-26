# -*- coding: utf-8 -*-
"""
CSE641 - Deep Learning - Assignment 1

PART I: Perceptron Training Algorithm - Question 2

@author: Shyama Sastha
"""

# imports
#import tensorflow as tf
from numpy.random import random

"""
Q2 a
Implementation of Madeleine learning algorithm for f(x1, x2)
# Points to be noted:
    # There are 2 inputs
    # 12 neurons in the first hidden layer
    # 5 neurons in the second hidden layer
    # 1 neuron in the output layer classifying y as 0 or 1
    The following abbreviations are used throughout:
        # mn - Madaline Network
        # xn - number of inputs x
        # hn - number of hidden layers h
        # yn - number of outputs y in the output layer
        # n - neuron
        # delz - delta value calculated as (desired - predicted)
        # hl - hidden layer
        # w - weights
        # row - row of data sample
        # z - net input to a neuron Sum(wijxi) + b
        # l - layer
        # nlx - inputs for next layer
        # o - output of a neuron
        # d - desired output value
        # y - predicted output value
        # eta - learning rates
        # nE - number of epochs
        # e - error
        # E - sum of square errors between desired and predicted
"""

# Defining the activation function
def activation(y):
    if y >= 0:
      return 1
    else:
      return -1

# Defining the function to create a Madaline network with random weights
def init_mn(xn, hn1, hn2, yn):
    mn = []
    hl1 = [{'w':[random() for i in range(xn+1)], 'i': int(i/4) , 'f': 0, 'l': 0} for i in range(hn1)]
    mn.append(hl1)
    hl2 = [{'w':[random() for i in range(hn2+1)], 'i': i, 'f': 0, 'l': 1} for i in range(hn2)]
    mn.append(hl2)
    # setting the output layer weights as 1
    yl = [{'w':[random() for i in range(hn2+1)], 'i': 1, 'l': 2, 'f': 0}]
    mn.append(yl)
    return mn

# Defining the function to create net input  for each neuron
def zij(n, x):
    w = n['w']
    z = w[-1] # The bias is always set as 1
    for j in range(len(w)-1):
        z += w[j] * x[j]
    return z

# Defining the function for forward propagation of weights
def propf(mn, row):
    x = row[:-1]
    for i in range(len(mn)):
        l = mn[i]
        nlx = [] #input for next layer
        for n in l:
            if n['f'] != 1:
                n['z'] = zij(n, x,)
                n['o'] = activation(n['z'])
                nlx.append(n['o'])
            else:
                nlx.append(n['o'])
        x = nlx
    return x

# Defining the function to identify the neuron with smallest affine z value and flip its output
def zsafv(mn, row, d):
    flag = 0
    flipback(mn)
    while flag != 1:
        lzm = []
        for l in mn:
            lzm = [n for n in l if n['f'] != 1]
        mm = min(lzm, key=lambda x:x['z'])
        for n in mn[mm['l']]:
            if n['z'] == mm['z']:
                n['o'] = -1 * n['o']
                n['f'] = 1
        flag = (propf(mn, row)[0] == d)
    return mm


# Defining the function to flip back all the flag values
def flipback(mn):
    for i in range(len(mn)):
        for n in mn[i]:
            n['f'] = 0

# Defining the function for Updating weights
def updatew(mn, row, lr, d):
    n = zsafv(mn, row, d)
    i = n['l']
    x = row[:-1]
    if n['l'] != 1:
        x = [n['o'] for n in mn[i-1]]
    for k in range(len(x)):
        n['w'][k] += lr * (d - n['z']) * x[k]
        n['w'][-1] += lr * (d - n['z'])
    print('Weights updated successfully')

# Defining the function to train the Madaline network for a nE epochs
def train_mn(mn, ddata, lr, nE, yn):
    for epoch in range(nE):
        E = 0
        for row in data:
            output = propf(mn, row)[0]
            d = row[-1]
            E += 0.5 * (d-output)**2
            if d != output:
                updatew(mn, row, lr, d)
    print('\n')
    print('-> Final epoch= {}, learning rate= {}, Error= {}, Desired = {}, Predicted= {}'.format(epoch+1, lr, E, d, output))
    print('\n')

# Sample inputs for training
data = [[5, 1, 1], [5.5, 1.5, 1],
        [1, 1, -1], [3, 3 ,-1],
        [5, 3, -1], [5.5, 3.5, -1],
        [7, 3, -1], [9, 1, -1],
        [1, 5, 1], [1.5, 5.5, 1],
        [3, 5, -1], [3.5, 5.5, -1],
        [5, 5, 1], [5.5, 5.5, 1],
        [7, 5, -1], [7.5, 5.5, -1],
        [9, 5, 1], [9.5, 5.5, 1],
        [3, 7, -1], [1, 9, -1],
        [5, 7, -1], [5.5, 7.5, -1],
        [5, 9, 1], [5.5, 9.5, 1],
        [7, 7, -1], [9, 9, -1]]
xn = len(data[0]) - 1
yn = 1
mn = init_mn(xn, 16, 4, yn)
train_mn(mn, data, 0.1, 1000, yn)
for l in mn:
    print(l)
    print("\n")

for row in data:
    y = propf(mn, row)[0]
    print('Desired: {}, Predicted: {}' .format(row[-1], y))

"""

Q2 b No it is not possible compute f(x1, x2) in 2 neurons because there are multiple boundary conditions.
Every shaded area corresponds to 1 while there are unshaded regions inbetween.
This means a minimum of 6 vertical boundaries and 6 horizontal boundaries are required to identify the right regions.
Then to actually figure out the bounds for each square, 5 more neurons are needed.
Because 4 of the 12 neurons from the first layer will bound each box through AND condition and there are 5 boxes.
And to Sum it all up, a neuron is required to make up for the OR condition from the 5 outputs to get a final outcome.

"""
