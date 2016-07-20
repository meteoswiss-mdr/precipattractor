#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as md
from matplotlib.collections import LineCollection
import pylab

from scipy import stats
import datetime
import glob
import numpy.ma as ma

import utils as ut
import importlib
importlib.reload(ut)

fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"

################# DEFAULT ARGS #########################
product = 'AQC'
timeStartStr = '201601310000' # args.start
inBaseDir = '/store/msrad/radar/precip_attractor/data/' #'/scratch/lforesti/data/'
outBaseDir = '/users/lforesti/results/'
timeStatsHours = 24
timeAccumMin = 5

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Plot radar rainfall field statistics.')
parser.add_argument('-start', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-WOLS', default=0, type=int,help='Whether to use the weighted ordinary leas squares or not in the fitting of the power spectrum.')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
args = parser.parse_args()

product = args.product
if (int(args.start) > int(args.end)):
    print('Time end should be after time start')
    sys.exit(1)

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start
    timeEndStr = args.end

timeStart = ut.timestring2datetime(timeStartStr)
timeEnd = ut.timestring2datetime(timeEndStr)

timeAccumMinStr = '%05i' % timeAccumMin
timeAccum24hStr = '%05i' % (timeStatsHours*60)

############### OPEN FILES WITH STATS
arrayStats = ut.csv_list2array(timeStart, timeEnd, inBaseDir, product, timeStatsHours, args.WOLS, args.minR)

if len(arrayStats) == 0:
    print("No data found in CSV files.")
    sys.exit(1)
############## VARIABLES TO PLOT ##########################
var1 = 'WAR' #r'$\beta_1$'
col1 = arrayStats[:,1]
var2 = r'$\beta_1$'
col2 = -arrayStats[:,10]
warLims = [0,50]
beta1Lims = [1.7,3.0]
###########################################################

if (var1 == r'$\beta_1$'):
    var1str = 'beta1'
elif (var1 == r'$\beta_2$'):
    var1str = 'beta2'
else:
    var1str = var1

if (var2 == r'$\beta_1$'):
    var2str = 'beta1'
elif (var2 == r'$\beta_2$') :
    var2str = 'beta2'
else:
    var2str = var2

############### GENERATE TIME STAMPS
timeStamps = np.array(arrayStats[:,0],dtype=int)
timeStampsStr = np.array(list(map(str,timeStamps)))

timeStampsDt = []
for t in range(0,timeStampsStr.shape[0]):
    timeStampsDt.append(ut.mkdate(timeStampsStr[t]))

############### PLOTTING TIME SERIES
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax1=plt.gca()
xfmt = md.DateFormatter('%H:%M') #'%Y-%m-%d %H:%M:%S'
ax1.xaxis.set_major_formatter(xfmt)

# First axis
ax1.plot(timeStampsDt, col1, 'b-')
ax1.set_xlabel('Time')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel(var1, color='b')
# Set axis limits
if var1 == 'WAR':
    ax1.set_ylim(warLims)
if var1 == r'$\beta_1$':
    ax1.set_ylim(beta1Lims)
for tl in ax1.get_yticklabels():
    tl.set_color('b')

# Second axis
ax2 = ax1.twinx()
ax2.plot(timeStampsDt, col2, 'r-')
ax2.set_ylabel(var2, color='r')
# Set axis limits
if var2 == 'WAR':
    ax2.set_ylim(warLims)
if var2 == r'$\beta_1$':
    ax2.set_ylim(beta1Lims)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
# Save
titleStr = product + ': ' + str(timeStampsDt[0]) + ' - ' + str(timeStampsDt[len(timeStampsDt)-1])
plt.title(titleStr, fontsize=14)
plt.xlabel('Time')

fileName = outBaseDir + product + timeStartStr + '-' + timeEndStr + '0_00005_timeSeries_' + var1str + '-' + var2str + '.png'
print('Saving: ',fileName)
plt.savefig(fileName)

############### PLOTTING SCATTERPLOTS  
plt.close("all")
plt.scatter(col1,col2)
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(titleStr)
# Compute linear regression
slope, intercept, r, p_value, std_err = stats.linregress(col1,col2)

plt.plot([np.min(col1), np.max(col1)], [intercept + np.min(col1)*slope, intercept + np.max(col1)*slope])
rtxt = 'r = ' + str('%.3f' % r)
plt.text(np.max(col1), np.max(col2), rtxt)

# Set axis limits
if var1 == 'WAR':
    plt.xlim(warLims)
if var1 == r'$\beta_1$':
    plt.xlim(beta1Lims)
if var2 == 'WAR':
    plt.ylim(warLims)
if var2 == r'$\beta_1$':
    plt.ylim(beta1Lims)
# Save
fileName = outBaseDir + product + timeStartStr + '-' + timeEndStr + '0_00005_scatter_' + var1str + '-' + var2str + '.png'
print('Saving: ',fileName)
plt.savefig(fileName)

############### PLOTTING SINGLE ATTRACTOR
# get hour from day
hoursList = []
for t in range(0,len(timeStampsDt)):
    secondsFromStart = (timeStampsDt[t] - timeStampsDt[0]).total_seconds()
    hoursList.append(secondsFromStart/3600)

# plot attractor
# plt.close("all")
# axSc = plt.scatter(col1, col2, c=np.array(hoursList), cmap=plt.get_cmap('hsv'))
# plt.xlabel(var1)
# plt.ylabel(var2)
# cbar = plt.colorbar(axSc)
# cbar.ax.set_ylabel('Hours from midnight')

##### Plot collection
plt.close("all")
points = np.array([col1, col2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Compute duration of event for colour scale
periodDuration = timeStampsDt[0] - timeStampsDt[len(timeStampsDt)-1]
periodDurationHours = np.abs(periodDuration.total_seconds())/3600

lc = LineCollection(segments, cmap=plt.get_cmap('hsv'),
                    norm=plt.Normalize(0, periodDurationHours))
lc.set_array(np.array(hoursList))
lc.set_linewidth(3)
fig2 = plt.figure()
ax = plt.gca().add_collection(lc)

plt.xlim(np.min(col1),np.max(col1))
plt.ylim(np.min(col2),np.max(col2))
cbar = plt.colorbar(ax)
cbar.ax.set_ylabel('Hours from start of event')
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(titleStr)
# Set axis limits
if var1 == 'WAR':
    plt.xlim(warLims)
if var1 == r'$\beta_1$':
    plt.xlim(beta1Lims)
if var2 == 'WAR':
    plt.ylim(warLims)
if var2 == r'$\beta_1$':
    plt.ylim(beta1Lims)

fileName = outBaseDir + product + timeStartStr + '-' + timeEndStr + '0_00005_attractor_' + var1str + '-' + var2str + '.png'
print('Saving: ',fileName)
plt.savefig(fileName)
