# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 09:12:47 2018

@author: jrobc
"""
#This code is for applying the fractional laplacian operator to a smaller image.
import numpy as np
import scipy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import sparse as SP
from datetime import datetime
from PIL import Image



startTime = datetime.now()

#gray = np.zeros((10,10))
#for i in range (0,10):
#    for j in range(1,9):
#        if i > 0:
#            gray[i,j]=1
#        if i == 9:
#            gray[i,j]=0
            
#imgplt = plt.imshow(gray, cmap = plt.get_cmap('gray_r'))            
            

img=mpimg.imread('stinkbugssmall.png')  #This block reads the image and converts to grayscale.
r, g, b =img[:,:,0], img[:,:,1], img[:,:,2]
gray = 20.0/61*r + 40.0/61*g +1.0/61*b 
n, m = np.shape(gray) #n rows, m columns
gray = np.reshape(gray,(n*m,1))

#Creating the laplacian operator.
#These create the base vectors for the Laplacian 
one = np.ones(n*m) 
four = -4*one #Base vector for diagonal.
oneoff = np.ones(n*m-1) #Base vector for the first sub and super diagonals.
off = np.ones(n*m-m) #Base vector for the m^th sub and super diagonals.
#The balancing pattern that shows that every pixel on the edge only touches at most 3 other pixels.
c = oneoff[m-1::m]
c[:] = 0
#the pattern that shows that the top row of pixels only touches at most 3 other pixels.
b = four[1:m]
b[:]=-3
#The pattern that shows the bottom row of pixels only touches at most 3 other pixels.
b = four[n*m-m-1:n*m-1]
b[:] = -3
#The pattern that shows the left column of pixels only touches at most 3 other pixels.
b = four[m::m]
b[:] = -3
#The pattern that shows the right column of pixels only touches at most 3 other pixels.
b = four[m-1::m]
b[:]=-3
#Correcting the four corner pixels to only touch two other pixels.
four[0]=-2
four[n*m-1]=-2
four[m-1]=-2
four[n*m-m]=-2
#Putting the correct 5 diagonals into a vector of diagonals.
diagonals = [four, oneoff, oneoff, off, off]
Lap = SP.diags(diagonals,[0,1,-1,-m,m]).toarray()
#Converting the laplacian operator to the schur form decomposition.
Z, Q = sp.linalg.schur(Lap)
Z.flags.writeable = True
Z = -Z
Z= np.diag(Z)
zero = np.argmin(Z)
Z.flags.writeable = True
Z[zero]=0
C = Q.T
fig = plt.figure()


#General eigenvalue decomp.
#D, V = LA.eig(Lap)
#S = -D
#zero = np.argmin(S)
#S[zero] = 0
#fig = plt.figure()
#C = LA.inv(V)
#Setting up an array to make a color map.
fractionals = []

for i in range(0,12):
    
    #This is for using general eigenvalue decomp.
    #T=S
    #T=T**(1/5.0)
    #T = np.diag(T)
    #Laps = -V.dot(T.dot(C))
    #Laps = Laps.real   #The eigenvalue solver is giving a very small imaginary portion, should be zero, as the matrix is symmetric.
    #Laps = np.dot(Laps,gray)  #Applying the fractional Laplacian to the grayscale.
    #Laps = np.reshape(Laps,(n,m)) 
    
    #If I use schur form decomp.
    S=Z
    S=S**(i/5.0)
    S = np.diag(S)
    Laps = -Q.dot(S.dot(C))
    Laps = Laps.real   #The eigenvalue solver is giving a very small imaginary portion, should be zero, as the matrix is symmetric.
    Laps = np.dot(Laps,gray)  #Applying the fractional Laplacian to the grayscale.
    Laps = np.reshape(Laps,(n,m)) 
    
    #Storing each of the instances of the fractional laplacian.
    fractionals.append(Laps)
    
#Creating the right normalization factor for the fractional laplacians to plot.   
small = np.min(fractionals)
large = np.max(fractionals)

for i in range(0,12):
    nLaps = fractionals[i]
    fig.add_subplot(3,4,i+1)
    #Uncomment one of the next two lines for the code to plot.  The first normalizes.
    imgplt = plt.imshow(nLaps, cmap = plt.get_cmap('gray'),vmin=small, vmax=large) 
    #imgplt = plt.imshow(nLaps, cmap = plt.get_cmap('gray_r'))
    #Plotting the images.
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(imgplt, cax=cbar_ax)

#The Below works, but is kind of uninspired.

print datetime.now() - startTime
