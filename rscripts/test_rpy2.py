import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import sys

robjects.r('''
       source('read_rzc.r')
''')

r_readRZC = robjects.globalenv['readRZC']

filename = '/scratch/ned/data/2016/16107/RZC161072100VL.801'

precip = r_readRZC(filename) # mm/h
precip = np.array(precip) 


plt.imshow(precip.T)
plt.colorbar()
plt.show()