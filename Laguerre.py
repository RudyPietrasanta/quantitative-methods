# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:47:49 2025

Laguerre integration

@author: rodol
"""
import math
import numpy as np


n =  10   #Degree of the polynomial

def LaguerreCoeff(n):
    c = np.zeros(n+1)
    for i in range(n+1):
        binomial = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
        c[i] = ((-1)**i)*binomial/math.factorial(i)
    return np.flip(c)


cn = LaguerreCoeff(n)                   #Coefficiants of Laguerre degree n
x = np.roots(cn)                        #Nodes
cm = LaguerreCoeff(n+1)                 #Coefficients of Laguerre degree n+1
w = x/((n+1)*np.polyval(cm,x))**2       #Weights

def f(x):                               #Function to evalute the integral of against exp(-x)
    return x**2

I = 0       
for i in range(n):
    I += w[i]*f(x[i])                   #I = sum_i (w_i*f(x_i))

