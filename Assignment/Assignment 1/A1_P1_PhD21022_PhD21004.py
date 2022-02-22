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

"""
Implementation of PTA for AND, OR and NOT gates
# Points to be noted:
    # The iterations will be much less when the learning rate is set to 1.
    # The reason for not setting the learning rate to 1 is to find the bound
    # Cannot compute bound equation if w2 becomes 0 due to divide by zero error
    # It can be seen from the XOR graphs that after a few tries, the perceptron
      gives up and assumes two boundaries instead of a single bound.
"""

# step function for prediction
def prediction(y_hat):
  if y_hat >= 0:
    return 1
  else:
    return 0

# Decision boundary line is given by x1w1 + x2w2 + b = 0
# Reforming this equation to get x2 = -(x1w1 + b)/w2 as boundary function
def bound(x1, w1, b, w2):
    if w2 != 0:
        return (-1 * ((x1 * w1) + b)/w2)
    else:
        return (-1 * ((x1 * w1) + b))

# Decision point is given by x1w1 + b = 0
# Reforming this equation to get x1 = -b/w1 as predicted point
def boundN(w1, b):
    if w1 != 0:
        return (-1 * b/w1)
    else:
        return (-1 * b)

# Defining the plot function for 2 inputs
def plotbound(X1, X2, pp, y, cc):
    xp, yp = x.T
    colormap = np.array(['r', 'g'])
    plt.scatter(xp, yp, s=20, c=colormap[y])
    plt.plot(X1,X2, color=cc)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Updating Boundary updating weights')
    plt.savefig(pp, format='pdf')

# Defining the plot function for 1 input
def plotboundN(X1, pp, cc):
    plt.plot(0, 0, marker="o", markersize=20, color="g")
    plt.plot(1, 1, marker="o", markersize=20, color="r")
    plt.plot(X1, X1, color=cc)
    plt.xlabel('X1')
    plt.ylabel('b')
    plt.title('Updating Boundary updating weights')
    plt.savefig(pp, format='pdf')

# Defining the function to train the perceptron and plot the boundaries
def train(x, y, pp, G):
    # Initialize random values of weights for w1 and w2
    w = np.random.randint(3, size=(1, 3))
    w1 = w[0][0]
    w2 = w[0][1]
    bias = w[0][2]

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

    # Plotting the initial boundary
    plotbound(X1, X2, pp, y, "black")

    # The max number of iterations is given to make sure the loop exits
    # The PTA runs till convergence for a linearly separable problem
    max = 100
    itr = 1
    lr = 0.2
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
        plotbound(X1, X2, pp, y, "black")
        E = np.sum(error) # Sum of errors
        itr = itr + 1 # Total number of iterations
    plotbound(X1, X2, pp, y, "green")
    wandb.append([w1, w2, bias])
    plt.clf()
    print("Final number of iterations for the {} PTA: {}".format(G, itr+1))
    print("Inital and Final weights & bias for the {} PTA: {}".format(G, wandb))

# Defining the training function for NOT gate
def trainN(x, y, pp, G):
    # Initialize random values of weights for w1 and w2
    w = np.random.randint(3, size=(1, 2))
    w1 = w[0][0]
    bias = w[0][1]

    # Find out the error for the first iteration
    X1 = [None] * 2
    error = np.array([0,0])
    for i in range(len(x)):
            y_hat = prediction(np.dot(w1 , x[i])  + bias)
            error[i] = y[i] - y_hat
            X1[i] = boundN(w1, bias)
    E = np.sum(error)

    # Plotting the initial prediction
    plotboundN(X1, pp, "black")

    # The max number of iterations is given to make sure the loop exits
    # The PTA runs till convergence for a linearly separable problem
    max = 50
    itr = 1
    lr = 0.2
    wandb = [[w1, bias]]
    while i < max & E != 0:
        for i in range(len(x)):
            y_hat = prediction(np.dot(w1, x[i])  + bias)
            error[i] = y[i] - y_hat
            w1 = w1 + lr * error[i] * x[i]
            bias = bias + lr * error[i]
            X1[i] = boundN(w1, bias)
        plotboundN(X1, pp, "black")
        E = np.sum(error) # Sum of errors
        itr = itr + 1 # Total number of iterations
    plotboundN(X1, pp, "green")
    wandb.append([w1, bias])
    plt.clf()
    print("Final number of iterations for the {} PTA: {}".format(G, itr+1))
    print("Inital and Final weights & bias for the {} PTA: {}".format(G, wandb))


# Inputs and Expected outputs for AND, OR and NOT gates
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y_AND = np.array([0, 0, 0, 1])
y_OR = np.array([0, 1, 1, 1])
x_NOT = np.array([0, 1])
y_NOT = np.array([1, 0])
y_XOR = np.array([0, 1, 1, 0])

# Train models and create plots for AND - Q1, a & b
pp_AND = PdfPages('Iteration_plots_AND.pdf') # Q1, b
train(x, y_AND, pp_AND, "AND") # Q1, a, i
pp_AND.close()

# Train models and create plots for OR - Q1, a & b
pp_OR = PdfPages('Iteration_plots_OR.pdf') # Q1, b
train(x, y_OR, pp_OR, "OR") # Q1, a, i
pp_OR.close()

# Train models and create plots for NOT - Q1, a & b
pp_NOT = PdfPages('Iteration_plots_NOT.pdf') # Q1, b
trainN(x_NOT, y_NOT, pp_NOT, "NOT") # Q1, a, ii
pp_NOT.close()

# Train models and create plots for XOR - Q1, c
# The perceptron gives up after a few iterations
pp_XOR = PdfPages('Iteration_plots_XOR.pdf') # Q1, c
train(x, y_XOR, pp_XOR, "XOR") # Q1, c
pp_XOR.close()


"""
Implementation of Madeleine learning algorithm for f(x1, x2)
# Points to be noted:
    #
"""