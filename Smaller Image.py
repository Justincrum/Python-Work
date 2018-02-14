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
from datetime import datetime

startTime = datetime.now()

img=mpimg.imread('stinkbugsmall.png')  #This block reads the image and converts to grayscale.
r, g, b =img[:,:,0], img[:,:,1], img[:,:,2]
gray = 20.0/61*r + 40.0/61*g +1.0/61*b
n, m = np.shape(gray)
gray = np.reshape(gray,(n*m,1))
#Creating the laplacian operator.
one = np.ones(n*m)
four = -4*one

oneoff = np.ones(n*m-1)
four[0]=-2
b = four[1:m]
b[:]=-3
b = four[n*m-m-2:n*m-1]
b[:] = -3
four[n*m-1]=-2
off = np.ones(n*m-m)
diagonals = [four, oneoff, oneoff, off, off]
Lap = SP.diags(diagonals,[0,1,-1,-m,m]).toarray()
#Converting the laplacian operator to the eigenvalue decomposition.
D, V = LA.eig(Lap)

fig = plt.figure()


for i in range(0,6):
    T=D
    T=T**i/5
    T = np.diag(T)
    Laps = -V.dot(T.dot(LA.inv(V)))
    Laps = Laps.real   #The eigenvalue solver is giving a very small imaginary portion, should be zero, as the matrix is symmetric.
    Laps = np.dot(Laps,gray)  #Applying the fractional Laplacian to the grayscale.
    Laps = np.reshape(Laps,(100,100))  
    fig.add_subplot(2,3,i+1)
    imgplt = plt.imshow(Laps, cmap = plt.get_cmap('gray'))  #Plotting the image.

#The Below works, but is kind of uninspired.

#Laps = -V.dot(S.dot(LA.inv(V)))
#Laps = Laps.real   #The eigenvalue solver is giving a very small imaginary portion, should be zero, as the matrix is symmetric.
#Laps = np.dot(Laps,gray)  #Applying the fractional Laplacian to the grayscale.
#Laps = np.reshape(Laps,(100,100))  

#imgplt = plt.imshow(Laps, cmap = plt.get_cmap('gray'))  #Plotting the image.
print datetime.now() - startTime
