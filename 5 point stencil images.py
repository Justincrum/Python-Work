# -*- coding: utf-8 -*-
"""
Spyder Editor
This file will compute the 5 point stencil discrete Laplacian in 2-d.
This is a temporary script file.
""" 
import numpy as np
from numpy import linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import *
from scipy import sparse as SS


img=mpimg.imread('stinkbug.png')  #This block reads the image and converts to grayscale.
r, g, b =img[:,:,0], img[:,:,1], img[:,:,2]
gray = 20.0/61*r + 40.0/61*g +1.0/61*b
n, m = np.shape(gray)
gray = np.reshape(gray,(n*m,1))
#imgplot = plt.imshow(img, cmap = plt.get_cmap('gray'))
size=n*m
one=np.ones(size)
foff=np.ones(size-m+2)
four=-4*one
oneoff=np.ones(size-1)
diagonals = [four, oneoff, oneoff, foff, foff]
Lap = SS.diags(diagonals,[0,1,-1,m-2,-m+2])  #Discrete Laplacian matrix, made as a sparse diagonal array.


D, V = SS.linalg.eigs(Lap)
#S = np.diag(D)
#S = -S
#S = S**.3 #Finding the s power of the diagonal to create the fractional laplacian.
#Converting discrete laplacian A to A^s.
#As = -V.dot(S.dot(LA.inv(V)))
#Laps = np.dot(As,gray)
#Laps = np.reshape(Laps,(375,500))
#imgplot = plt.imshow(Laps, cmap = plt.get_cmap('gray'))
#plotting the picture of the discrete fractional laplacian applied to the original grayscale.
 
