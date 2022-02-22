# -*- coding: utf-8 -*-
"""
CSE641 - Deep Learning - Assignment 1

PART I: Perceptron Training Algorithm

@author: Shyama Sastha
"""

# imports
#import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
%matplotlib inline

# step function for prediction 
def prediction(y_hat):
  if y_hat >= 0:
    return 1
  else:
    return 0

"""
Implementation of PTA for AND GATE
""" 

# Initialize random values of weights for w1 and w2
w = np.random.randint(4, size=(1, 3))
w1 = w[0][0]
w2 = w[0][1]
bias = w[0][2]

# Inputs and expected outputs for AND gate
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = x[:,0] * x[:,1]

# Decision boundary line is given by x1w1 + x2w2 + b = 0
# Reforming this equation to get x2 = -(x1w1 + b)/w2
def bound(x1, w1, b, w2):
  return (-1 * ((x1 * w1) + b)/w2)

# Find out the error for the first iteration
X1 = [None] * 4
X2 = [None] * 4
error = np.array([0,0,0,0])
for i in range(len(x)):
        y_hat = prediction(np.dot(np.array([w1, w2]) , x[i])  + bias)
        error[i] = y[i] - y_hat
        X1[i] = x[i][0]
        X2[i] = bound(x[i][0], w1, bias, w2)
E = np.sum(error)

# plotting the initial boundary
xp, yp = x.T
pp = PdfPages('Iteration plots.pdf')
plt.scatter(xp, yp)
plt.plot(X1,X2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Inital Boundary with randomized initial weights')
plt.savefig(pp, format='pdf')

# The max number of iterations is given to make sure the loop exits
# The PTA runs till convergence (which is a given in a linearly separable problem)

max = 100
itr = 1
lr = 0.1
wandb = [[w1, w2, bias]]
while i < max & E != 0:
    for i in range(len(x)):
        y_hat = prediction(np.dot(np.array([w1, w2]) , x[i])  + bias)
        error[i] = y[i] - y_hat
        w1 = w1 + lr * error[i] * x[i][0]
        w2 = w2 + lr * error[i] * x[i][1]
        bias = bias + lr * error[i]
        X1[i] = x[i][0]
        X2[i] = bound(x[i][0], w1, bias, w2)
    plt.scatter(xp, yp)
    plt.plot(X1,X2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Boundary progression with updated weights')
    plt.savefig(pp, format="pdf")
    E = np.sum(error) # Sum of errors
    itr = itr + 1 # Total number of iterations
wandb.append([w1, w2, bias])
pp.close()
print("Final number of iterations: ", itr-1)
print("Final error: ", E)
print("Final weights and the bias are ", wandb[1])
   