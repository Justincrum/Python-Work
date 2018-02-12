# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 09:12:47 2018

@author: jrobc
"""
#This code is for applying the fractional laplacian operator to a smaller image.
import numpy as np
from numpy import linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import *
from scipy import sparse as SP

img=mpimg.imread('stinkbugsmall.png')  #This block reads the image and converts to grayscale.
r, g, b =img[:,:,0], img[:,:,1], img[:,:,2]
gray = 20.0/61*r + 40.0/61*g +1.0/61*b
n, m = np.shape(gray)
gray = np.reshape(gray,(n*m,1))
#Creating the laplacian operator.
one = np.ones(n*m)
four = -4*one
oneoff = np.ones(n*m-1)
b = oneoff[2::3]
b[:]=0
off = np.ones(n*m-m+1)
diagonals = [four, oneoff, oneoff, off, off]
Lap = SP.diags(diagonals,[0,1,-1,-m+1,m-1]).toarray()
#Converting the laplacian operator to the eigenvalue decomposition.
D, V = LA.eig(Lap)
S = np.diag(D)
S = -S
S = S**.3  #Raising the eigenvalues to a fractional power.
Laps = -V.dot(S.dot(LA.inv(V)))
Laps = Laps.real   #The eigenvalue solver is giving a very small imaginary portion, should be zero, as the matrix is symmetric.
Laps = np.dot(Laps,gray)  #Applying the fractional Laplacian to the grayscale.
Laps = np.reshape(Laps,(100,100))  

imgplt = plt.imshow(Laps, cmap = plt.get_cmap('gray'))  #Plotting the 