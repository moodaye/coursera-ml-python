# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:46:44 2019

@author: Rajiv
"""

import numpy as np
from pylab import *

def computeCost(X, y, theta):
    o1 = ones(x.size)
    X = np.vstack((o1, x))
    h=xtheta.__matmul__(X)
    error = h - y
    error_sqr = error * error
#    print(sum(error_sqr)/2/size(x))
    return sum(error_sqr)/2/size(x)

def gradientDescent(X, y, theta, alpha, iters):
    # step 0: m = length(y)
    m = size(y)
    print(m)
    # step 0: J_history = zeros(num_iters, 1)
    J_history = zeros(iters)
    print(J_history)
    
    o1 = ones(x.size)
    X = np.vstack((o1, x))

    # step 1: for iter = 1 : num_iters
    for i in range(1, iters):
    # step 2: h = X * theta
        h = theta.__matmul__(X)
    # step 3: error = h - y
        err = h - y
    # step 4: scale1 = X' * error
        scale1 = err.__matmul__(X.transpose())
    # step 5: theta = theta - (scale1.*(alpha / m))
        theta = theta - (scale1*(alpha / m))
        J_history[i] = computeCost(X, y, theta)
    return theta
    
d = np.loadtxt("ex1data1.txt", delimiter=",")
x=d[:,0]
y=d[:,1]

scatter(x,y, c='red', marker='x')
xlabel("Population of city in 10,000s")
ylabel("Profit in $10,000s")

xtheta=np.array([[0,0]])
print(computeCost(x, y, xtheta))
xtheta=np.array([[-1,2]])
print(computeCost(x, y, xtheta))

theta = gradientDescent(x, y, xtheta, 1, 11)
print(theta)

