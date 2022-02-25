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
      return 0

# Defining the function to create a Madaline network with random weights
def init_mn(xn, hn1, hn2, yn):
    mn = []
    hl1 = [{'w':[random() for i in range(xn + 1)]} for i in range(hn1)]
    mn.append(hl1)
    hl2 = [{'w':[random() for i in range(hn1 + 1)]} for i in range(hn2)]
    mn.append(hl2)
    # setting the output layer weights as 1
    yl = [{'w':[1 for i in range(hn2 + 1)]}]
    mn.append(yl)
    return mn

# Defining the function to create net input  for each neuron
def zij(w, x):
    z = w[-1] #the final random weight is chosen as bias
    for i in range(len(w)-1):
        z += w[i] * x[i]
    return z

# Defining the function for forward propagation of weights
def propf(mn, row):
    x = row
    for i in range(len(mn)):
        l = mn[i]
        nlx = [] #input for next layer
        for n in l:
            n['z'] = zij(n['w'], x)
            n['o'] = activation(n['z'])
            nlx.append(n['o'])
        x = nlx
    return x

#Defining the function for finding the neuron with smallest z value
def zmin(l, z):
    for n in l:
        if z == n['z']:
            return n

# Defining the function for Updating weights
def updatew(mn, row, lr):
    for i in reversed(range(len(mn)-1)):
        j = i-1
        x = row[:-1]
        d = row[-1]
        for n in mn[i]:
            if d != n['o']:
                z_min = min(mn[j], key=lambda x:x['z'])
                m = zmin(mn[j], z_min['z'])
                if j != 0:
                    x = [n['o'] for n in mn[j]]
                for k in range(len(x)):
                    m['w'][k] += lr * n['del'] * x[k]
                m['w'][-1] += lr * n['del']

# Defining the function for backward propagation of errors
def propb(mn, d):
    for i in reversed(range(len(mn)-1)):
        l = mn[i]
        for j in range(len(l)):
            n = l[j]
            n['del'] = (d - n['z'])

# Defining the function to train the Madaline network for a nE epochs
def train_mn(mn, ddata, lr, nE, yn):
    for epoch in range(nE):
        E = 0
        for row in data:
            output = propf(mn, row)[0]
            d = row[-1]
            E += (d-output)**2
            propb(mn, d)
            updatew(mn, row, lr)
        print('-> epoch= {}, lrate= {}, Error= {}, Predicted= {}'.format(epoch, lr, E, output))
    print('\n')

# Sample inputs for training
data = [[1, 1, 0], [3, 3 ,0],
        [5, 1, 1], [5.5, 1.5, 1],
        [5, 3, 0], [5.5, 3.5, 0],
        [7, 3, 0], [9, 1, 0],
        [1, 5, 1], [1.5, 5.5, 1],
        [3, 5, 0], [3.5, 5.5, 0],
        [5, 5, 1], [5.5, 5.5, 1],
        [7, 5, 0], [7.5, 5.5, 0],
        [9, 5, 1], [9.5, 5.5, 1],
        [3, 7, 0], [1, 9, 0],
        [5, 7, 0], [5.5, 7.5, 0],
        [5, 9, 1], [5.5, 9.5, 1],
        [7, 7, 0], [9, 9, 0]]
xn = len(data[0]) - 1
yn = 1
mn = init_mn(xn, 12, 5, yn)
train_mn(mn, data, 0.1, 10000, yn)
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


"""
# Initial idea for Adaline.
# First for every neuron in the last layer
# Then process them layer by layer
# Till you reach the first layer

# Defining function for backpropagation of error for one neuron
def Perceptron(X, W, b, d):
    # Initializing weights
    max =100
    eta = 0.1
    itr = 1
    error = 1
    z = np.dot(W, X) + b # This part can be made into a function
    while (itr < max and error == 1):
        for i in range(len(X)):
            for j in range(len(W)):
                y = activation(z)
                if y != d:
                    # using weight update rule of Adaline
                    W[i][j] = W[i][j] + (eta * (d - z) * X[i])
                    b[i][j] = b[i][j] + (eta * (d - z))
"""
