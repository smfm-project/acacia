################################################################################
################################################################################
# Import external and internal modules

from __future__ import division

import os
import numpy as np
import pandas as bb
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import shapefile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
import scipy.spatial
import scipy.cluster
import shapely.geometry
import shapefile
import skimage.morphology
import sklearn.cluster
import sklearn.preprocessing

import scipy.ndimage as ndimage
from scipy.ndimage.measurements import label


import functions as fn


################################################################################
################################################################################
# test zone: the recursive function
def recfun (x, Points, indices, depth, out): # this works
    if depth != 0:
        print(np.count_nonzero(x)/x.size)
        for i in range(len(x)):
            indices = indices + (i,)
            if len(indices) == len(x.shape):

                x[indices] = infun(x, indices, Points, out)

            recfun (x, Points, indices, depth-1, out)
            indices = indices [:-1]
    return x

# test zone: the inside function
def infun (x, indices, points, out):
    cellsize = 1. / len(x)

    cellminis = np.zeros(len(indices)); cellmaxis = np.zeros(len(indices))
    for i in range(len(cellminis)):
        cellminis[i] = indices[i] * cellsize; cellmaxis[i] = (indices[i]+1) * cellsize

    A = points - cellminis; B = cellmaxis - points

    elements = []
    for i in range(len(A)):
        if min(A.iloc[i]) >= 0 and min(B.iloc[i]) >= 0:
            elements.append(A.index.values[i])
    if out == 'selection':
        if len(elements) >= 1:
            val = np.random.choice(elements)
        else:
            val = 0
    elif out == 'density':
        val  = len(elements)

    return val



################################################################################
################################################################################
# Import and data
OUT = '/home/willgoodwin/NPIF/Tool4/acacia/acacia_rcmrd/data/'

with open(OUT+'Classified_df.pkl', 'rb') as handle:
    bb, maxies = pickle.load(handle)

with open(OUT+'Validation_df.pkl', 'rb') as handle:
    vv = pickle.load(handle)

# S is the number of subdivisions of the grid in each dimension
S = 2

################################################################################
################################################################################
# prepare data
bb = bb.set_index('ChangeID')
clusters = bb['clusters']
bb = bb.drop(['clusters'], axis=1)

################################################################################
################################################################################
# generate grid
dimsizes = S * np.ones(len(bb.columns.values))
grid = np.zeros(dimsizes.astype(int))

bb['clusters'] = clusters

print (grid.size)

for c in range(0,max(clusters)):
    bbc = bb.loc[bb['clusters'] == c]

    bbc = bbc.drop(['clusters'], axis=1)
    selgrid = recfun (grid, bbc, (), len(grid.shape), 'selection')

    print (selgrid)


quit()









#fn.fig1 (clusters, cols, bb, OUT)
################################################################################
################################################################################
# Import data
R = range(0, max(clusters))

R2 = [10]

fn.figscatter (clusters, cols, bb, OUT, R)

fn.figscatterval (clusters, cols, bb, vv, OUT, R, maxies)
print (range(0, max(clusters)))


quit()
################################################################################
################################################################################
# run the recursive function


input = np.asarray([[[0,0],
                    [0,0]],
                   [[-0,-0],
                    [-0,-0]]])
ini_depth = len(input.shape)

point_coords = [(0.1, 0.1, 0.1), (0.2, 0.1, 0.2), (0.1, 0.8, 0.8)]

Points = bb.DataFrame(point_coords)
Points.index = np.arange(1, len(Points) + 1)
print (Points)

print('\nlaunching function\n')

X = recfun (input, (), ini_depth, 'selection')


print(X)



quit()







quit()
