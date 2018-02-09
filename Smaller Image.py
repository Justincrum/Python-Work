# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 09:12:47 2018

@author: jrobc
"""

import numpy as np
from numpy import linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import *
from scipy import sparse as SP

Z = np.random.rand(100,100)
Z = np.reshape(Z,10000)
#imgplt = plt.imshow(Z, cmap = plt.get_cmap('gray'))

one = np.ones(10000)
four = -4*one
oneoff = np.ones(9999)
b = oneoff[2::3]
b[:]=0
off = np.ones(9901)
diagonals = [four, oneoff, oneoff, off, off]
Lap = SP.diags(diagonals,[0,1,-1,-99,99]).toarray()
D, V = LA.eig(Lap)
S = np.diag(D)
S = -S
S = S**.3
Laps = -V.dot(S.dot(LA.inv(V)))
Laps = np.dot(Laps,Z)
Laps = np.reshape(Laps,(100,100))
Laps = Laps.real
imgplt = plt.imshow(Laps, cmap = plt.get_cmap('gray'))