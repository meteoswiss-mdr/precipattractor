#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import sys
import os

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as md
from matplotlib.collections import LineCollection
import pylab

from scipy import stats
import datetime
import glob
import numpy.ma as ma

import importlib

import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt

fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"

################# DEFAULT ARGS #########################
inBaseDir = '/store/msrad/radar/precip_attractor/data/' #'/scratch/lforesti/data/'
outBaseDir = '/users/lforesti/results/'
tmpBaseDir = '/scratch/lforesti/tmp/'
# Whether we used a variable scaling break 
plotHistScalingBreak = False

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Plot radar rainfall field statistics.')
parser.add_argument('-start', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-wols', default=0, type=int,help='Whether to use the weighted ordinary leas squares or not in the fitting of the power spectrum.')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
parser.add_argument('-minWAR', default=5, type=float,help='Minimum WAR threshold for plotting.')
parser.add_argument('-minCorrBeta', default=0.95, type=float,help='Minimum correlation coeff. for beta for plotting.')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-temp', default=5, type=int,help='Temporal sampling of the products [minutes].')
parser.add_argument('-format', default='netcdf', type=str,help='Format of the file containing the statistics [csv,netcdf].')
parser.add_argument('-refresh', default=0, type=int,help='Whether to refresh the binary .npy archive or not.')

args = parser.parse_args()

refreshArchive = bool(args.refresh)
print('Refresh archive:', refreshArchive)
product = args.product
timeAccumMin = args.accum
timeSampMin = args.temp

timeAccumMinStr = '%05i' % timeAccumMin
timeSampMinStr = '%05i' % timeSampMin

if (int(args.start) > int(args.end)):
    print('Time end should be after time start')
    sys.exit(1)

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start
    timeEndStr = args.end

timeStart = ti.timestring2datetime(timeStartStr)
timeEnd = ti.timestring2datetime(timeEndStr)

if plotHistScalingBreak:
    variableBreak = 1
else:
    variableBreak = 0
    
############### OPEN FILES WITH STATS
## Open single binary python file with stats to speed up (if it exists)
tmpArchiveFileName = tmpBaseDir + timeStartStr + '-' + timeEndStr + '_temporaryAttractor.npy'
tmpArchiveFileNameVariables = tmpBaseDir + timeStartStr + '-' + timeEndStr + '_temporaryAttractor_varNames.npy'
if (os.path.isfile(tmpArchiveFileName) == True) and (refreshArchive == False):
    arrayStats = np.load(tmpArchiveFileName)
    arrayStats = arrayStats.tolist()
    variableNames = np.load(tmpArchiveFileNameVariables)
    print('Loaded:', tmpArchiveFileName)
else:
    if args.format == 'csv':
        arrayStats, variableNames = io.csv_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
        product = product, timeAccumMin = timeSampMin, minR=args.minR, wols=args.wols, variableBreak=variableBreak)
    elif args.format == 'netcdf':
        arrayStats, variableNames = io.netcdf_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
        product = product, timeAccumMin = timeAccumMin, minR=args.minR, wols=args.wols, variableBreak=variableBreak)
    else:
        print('Please provide a valid file format.')
        sys.exit(1)

    # Check if there are enough data
    if (len(arrayStats) == 100) & (args.format == 'csv'):
        print("Not enough data found in CSV files.")
        sys.exit(1)
    if (len(arrayStats) < 100) & (args.format == 'netcdf'):
        print("No enough data found in NETCDF files.")
        sys.exit(1)

## Save data into a single binary bython file to speed up further analysis with same dataset
arrayData = []
if refreshArchive == True:
    np.save(tmpArchiveFileName, arrayStats)
    np.save(tmpArchiveFileNameVariables, variableNames)
    print('Saved:',tmpArchiveFileName)
    
# Generate list of datetime objects
timeIntList = dt.get_column_list(arrayStats, 0)
timeStampsDt = ti.timestring_array2datetime_array(timeIntList)

# Convert list of lists to numpy array
arrayStats = np.array(arrayStats)

# Check if there are data
print(len(arrayStats),' samples found.')
print('Variables from file: ', variableNames)

#################################################################################
####################### PLOTTING MULTIPLE ATTRACTORS in combinations of dimensions
varNamesRows = ['war','r_cmean', 'beta1', 'beta2','eccentricity']
varNamesCols = ['war','r_cmean', 'beta1', 'beta2']

#varNamesRows = ['eccentricity']
#varNamesCols = ['eccentricity', 'war','r_cmean', 'beta1', 'beta2']

warThreshold = args.minWAR
betaCorrThreshold = args.minCorrBeta

print('Variables for plotting: ', varNamesRows, ' vs. ', varNamesCols)

############### AXIS LIMITS
boolLogPlot = True
boolPowPlotEccentricity = False
if boolLogPlot:
    WARlims = [6,18] # [6,18]
    IMFlims = [-25,10] # [-20,5]
    MMlims = [-8,10] # [-20,5]    
else:
    WARlims = [warThreshold,60]
    IMFlims = [0.03, 3.0]
    MMlims = [0.5, 3.0] # [-20,5]
  
if boolPowPlotEccentricity:
    ecclims = [1,10]
else:
    ecclims = [0,1]

beta1lims = [1.2,2.8] #[1.6,2.8]
beta2lims = [2.2,4.3] #[3.2,4]

trajectoryPlot = 'sections' # 'lines' 'scatter' 'coloredlines' 'sections'
densityPlot = '2dhist'# 'kde' or '2dhist'
nrBinsX = 60
nrBinsY = 60

###############################################################################
# Generate labels for plotting
varNamesAll = varNamesRows + varNamesCols
varNamesAll = dt.unique(varNamesAll)

varLabels = []
for var in range(0, len(varNamesAll)):
    if varNamesAll[var] == 'war':
        varLabels.append('WAR')
    if varNamesAll[var] == 'r_mean':
        varLabels.append('IMF')
    if varNamesAll[var] == 'r_cmean':
        varLabels.append('MM')
    if varNamesAll[var] == 'beta1':
        varLabels.append(r'$\beta_1$')
    if varNamesAll[var] == 'beta2':
        varLabels.append(r'$\beta_2$')
    if varNamesAll[var] == 'eccentricity':
        varLabels.append('Eccentricity')

#####
# Get indices of variables
indicesVars = dt.get_variable_indices(varNamesAll, variableNames)

# Put indices into dictionary
dictIdx = dict(zip(varNamesAll, indicesVars))

# WAR threshold
boolWAR = (arrayStats[:,dictIdx['war']] >= warThreshold) 
# Beta correlation threshold
boolBetaCorr = (np.abs(arrayStats[:,dictIdx['beta1']+1]) >= np.abs(betaCorrThreshold)) & (np.abs(arrayStats[:,dictIdx['beta2']+1]) >= np.abs(betaCorrThreshold))
# Combination of thresholds
boolTot = np.logical_and(boolWAR == True, boolBetaCorr == True)

nrSamplesWAR = np.sum(boolWAR)
nrSamplesBetasWAR = np.sum(boolBetaCorr & boolWAR)
fractionValidBetas = 100*np.sum(boolBetaCorr & boolWAR)/nrSamplesWAR
print("Percentage valid betas: ", fmt1 % fractionValidBetas, " %")

############### Select subset of variables
varData = []
for var in range(0, len(varNamesAll)):
    varName = varNamesAll[var]
    if (varName == 'beta1') | (varName == 'beta2'):
        varData.append(-arrayStats[boolTot,dictIdx[varName]])
    else:
        varData.append(arrayStats[boolTot,dictIdx[varName]])

varData = np.array(varData).T

# Argument of maximum value (to select interesting cases)
# print(dictIdx)
# idxMax = np.argmax(arrayStats[:,6])
# maxWARtime = timeStampsDt[idxMax]
# print(maxWARtime, arrayStats[idxMax,:])
# sys.exit()

# Define variable indices of array subset
dictIdxSubset = dict(zip(varNamesAll, np.arange(len(varNamesAll))))

# Create array of data limits in correct order
varLimits = []
for var in range(0,len(varNamesAll)):
    varName = varNamesAll[var]
    if varName == 'war':
        dataLimits = WARlims
    if varName == 'r_mean':
        dataLimits = IMFlims
    if varName == 'r_cmean':
        dataLimits = MMlims
    if varName == 'beta1':
        dataLimits = beta1lims
    if varName == 'beta2':
        dataLimits = beta2lims
    if varName == 'eccentricity':
        dataLimits = ecclims        
    varLimits.append(dataLimits)
    
#axesLimits = np.array([ecclims, WARlims, MMlims, beta1lims, beta2lims])
axesLimits = np.array(varLimits)

# Define surfaces of section
minPercSec = 48
maxPercSec = 52
medianSectionStart = np.percentile(varData, minPercSec, axis=0)
medianSectionEnd = np.percentile(varData, maxPercSec, axis=0)

#sectionIntervals = np.array([[10,12],[1.0,1.1], [2.0, 2.2], [3.0,3.2]])
sectionIntervals = np.vstack((medianSectionStart,medianSectionEnd)).T

##### Select subset of array within given range
#print((arrayStats[arrayStats[:,dictIdx['eccentricity']] > 0.96, 0]).astype(int))
#sys.exit()
# boolData = (varData[:,2] > 1.45) & (varData[:,2] < 1.55) & (varData[:,3] > 3.0) & (varData[:,3] < 3.1)
# #varData = varData[boolData,:]
# timeStampsStr = timeStampsStr[boolTot]
# timeStampsSel = timeStampsStr[boolData]

# nrConsecFields = 0
# for i in range(0,len(timeStampsSel)-1):
    # timeDiff = (ti.timestring2datetime(timeStampsSel[i+1])-ti.timestring2datetime(timeStampsSel[i])).total_seconds()
    # if (timeDiff == 300.0):
        # nrConsecFields = nrConsecFields + 1
    # else:
        # nrConsecFields = 0
    
    # if (nrConsecFields == 12):
        # print(timeStampsSel[i-12])
################
############## HISTOGRAM SCALING BREAK
if plotHistScalingBreak:
    indexScalingVar = dt.get_variable_indices('scaling_break', variableNames)
    scalingBreak = arrayStats[boolTot,indexScalingVar]
    scaleBreaks = np.unique(scalingBreak)
    bins = np.hstack((scaleBreaks-1,50))

    counts, bins = np.histogram(scalingBreak, bins = bins)
    nrSamples = len(scalingBreak)
    counts = 100.0*counts/float(nrSamples)
    meanVal = np.nanmean(scalingBreak)
    medianVal = np.nanmedian(scalingBreak)
    stdVal = np.nanstd(scalingBreak)
    width = 0.4 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2.0

    # Plot hist
    axSb = plt.gca()
    print(scaleBreaks, counts)
    plt.bar(scaleBreaks, counts, align='center', width=width, color='blue', edgecolor='blue')
    textMedian = r'median = ' + str(int(medianVal))
    textMean = r'$\mu$ = ' + str("%0.2f" % meanVal)
    textStd = r'$\sigma$ = ' + str("%0.2f" % stdVal)
    plt.text(0.75, 0.95, textMedian, transform=axSb.transAxes, fontsize=14)
    plt.text(0.75, 0.91, textMean, transform=axSb.transAxes, fontsize=14)
    plt.text(0.75, 0.87, textStd, transform=axSb.transAxes, fontsize=14)

    maxPerc = 30
    plt.ylim([0, maxPerc])
    plt.xlabel('Scaling break [km]')
    plt.ylabel('Frequency [%]')

    #titleStr = 'Optimal scaling break \n' + product + ': ' + str(timeStampsDt[0]) + ' - ' + str(timeStampsDt[len(timeStampsDt)-1])
    titleStr = 'Optimal scaling break \n' + product + ': ' + str(timeStampsDt[0].year)
    plt.title(titleStr, fontsize=16)
    #plt.show()

    fileName = outBaseDir + product + timeStartStr + '-' + timeEndStr +  '0_' + \
    'Rgt' + str(args.minR) + '_WOLS' + str(args.wols) + '_00005_histScaleBreak_warGt' + str("%0.1f" % warThreshold) + '_' + timeAccumMinStr + '.png'
    print('Saving: ',fileName)
    plt.savefig(fileName, dpi=300)
    sys.exit()

####### PLOT SCALING BREAK VS ECCENTRICITY
ecc_idx = dictIdxSubset['eccentricity']
b1_idx = dictIdxSubset['beta1']
b2_idx = dictIdxSubset['beta2']

ecc = varData[:,ecc_idx]
eccDB = 10.0*np.log10(1-ecc)
b1 = varData[:,b1_idx]
b2 = varData[:,b2_idx]
diffBeta = b2-b1

nrBinsX = 50
nrBinsY = 50
xbins = np.linspace(np.min(eccDB), np.max(eccDB), nrBinsX)
ybins = np.linspace(np.min(diffBeta), np.max(diffBeta), nrBinsY)

# Compute histogram
counts, _, _ = np.histogram2d(eccDB, diffBeta, bins=(xbins, ybins))
nrSamples = len(eccDB)
counts[counts == 0] = np.nan
counts = counts/nrSamples*100
countsMask = ma.array(counts, mask = np.isnan(counts))
maxFreq = nrBinsX/(nrBinsX/0.30)

mpl.rc('xtick', labelsize=15) 
mpl.rc('ytick', labelsize=15) 

#### Plotting
# fig, ax = plt.subplots()
# newax = ax.twiny()

histIm = plt.pcolormesh(xbins, ybins, countsMask.T, vmax = maxFreq)                    
# plt.scatter(ecc, b2-b1)

# Axes
# newax.xaxis.set_ticks_position('bottom')
# xlocs = np.linspace(np.min(eccDB), np.max(eccDB), 10)
# xlabs = dt.from_dB(xlocs)+1
# print(xlabs)
# newax.set_frame_on(True)
# newax.patch.set_visible(False)
# newax.set_xticks(xlocs)
# newax.set_xticklabels(xlabs)

plt.xlabel('1-eccentricity [dB]', fontsize=22)
plt.ylabel(r'$\beta_2$-$\beta_1$', fontsize=22)
plt.show()
sys.exit()
    
#############PLOT ATTRACTOR
# Compute duration of event for colour scale
durationFromStart = timeStampsDt[0] - timeStampsDt[len(timeStampsDt)-1]
hoursFromStart = np.abs(durationFromStart.total_seconds())/3600
daysFromStart = np.abs(durationFromStart.total_seconds())/(3600*24)
if daysFromStart > 5:
    timeFromStart = daysFromStart
else:
    timeFromStart = hoursFromStart
   
# Adjust size of subplot for better visualization
nrRows = len(varNamesRows)
nrCols = len(varNamesCols)
if (nrRows == 1) and ((nrCols == 3) or (nrCols == 4)):
    nrRowsAdj = 2
    nrColsAdj = 2
elif (nrRows == 1) and (nrCols == 5):
    nrRowsAdj = 2
    nrColsAdj = 3
else:
    nrRowsAdj = nrRows
    nrColsAdj = nrCols
print('Number of subplots: ', nrRowsAdj, 'x', nrColsAdj)

# Generate figure
plt.close("all")
if nrRowsAdj == nrColsAdj:
    sizeFig = (11, 9.5)
else:
    sizeFig = (13, 9.5)

fig = plt.figure(figsize=sizeFig)
ax = fig.add_axes()
ax = fig.add_subplot(111)


if nrRows == nrCols:
    mpl.rc('xtick', labelsize=7) 
    mpl.rc('ytick', labelsize=7)
else:
    mpl.rc('xtick', labelsize=12) 
    mpl.rc('ytick', labelsize=12)
 
from matplotlib.ticker import FormatStrFormatter
p = 0

for row in range(0,nrRows):
    for col in range(0,nrCols): # loop first by changing the X axis variable and keeping the Y axis variable fixed
        ######## Select X and Y variable data
        # Y variable (kept fixed for a given row)
        varYname = varNamesRows[row]
        idxColVar = dictIdxSubset[varYname]
        if ((varNamesRows[row] == 'r_mean') or (varNamesRows[row] == 'r_cmean') or (varNamesRows[row] == 'war')) and boolLogPlot:
            if (varNamesRows[row] == 'r_mean') or (varNamesRows[row] == 'r_cmean'):
                offset = 0.005
            elif varNamesRows[row] == 'war':
                offset = 0.01
            else:
                offset = 0.0
            varY = 10*np.log10(varData[:,idxColVar] + offset)
            varYLab = varLabels[idxColVar] + ' [dB]'
        elif varNamesRows[row] == 'eccentricity' and boolPowPlotEccentricity:
                varY = 10**varData[:,idxColVar]
                varYLab = '10^(' + varLabels[idxColVar] + ')'
        else:
            varY = varData[:,idxColVar]
            varYLab = varLabels[idxColVar]
            
        # X variable (will vary along the columns of the plot)          
        varXname = varNamesCols[col]
        idxRowVar = dictIdxSubset[varXname]
        if ((varNamesCols[col] == 'r_mean') or (varNamesCols[col] == 'r_cmean') or (varNamesCols[col] == 'war')) and boolLogPlot:
            if (varNamesCols[col] == 'r_mean') or (varNamesCols[col] == 'r_cmean'):
                offset = 0.005
            elif varNamesCols[col] == 'war':
                offset = 0.01
            else:
                offset = 0.0
            varX = 10*np.log10(varData[:,idxRowVar] + offset)
            varXLab = varLabels[idxRowVar] + ' [dB]'
        elif varNamesCols[col] == 'eccentricity' and boolPowPlotEccentricity:
            varX = 10**varData[:,idxRowVar]
            varXLab = '10^(' + varLabels[idxRowVar] + ')'
        else:
            varX = varData[:,idxRowVar]
            varXLab = varLabels[idxRowVar]

        #########################################
        if nrRows == nrCols:
            ftSize = 8
        else:
            ftSize = 12
        # Plot number...
        p = p+1
        axSP = plt.subplot(nrRowsAdj, nrColsAdj, p)
        
        if (varYLab == r'$\beta_1$') or (varYLab == r'$\beta_2$') or (varYLab == 'IMF'):
            axSP.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if (varXLab == r'$\beta_1$') or (varXLab == r'$\beta_2$') or (varXLab == 'IMF'):
            axSP.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Compute liner regression
        slope, intercept, r, p_value, std_err = stats.linregress(varData[:,col], varData[:,row])
        xmin = np.min(varX)
        xmax = np.max(varX)
        ymin = np.min(varY)
        ymax = np.max(varY)
        
        ############ Plot attractor trajectories or sections
        if (row > col) and (nrRows == nrCols):
            print('Drawing ', trajectoryPlot,' at row, col=', row, ',',col, ' - ', varLabels[row], ' vs ', varLabels[col])
            titleStrSP = 'r=' + '%.2f' % r
            if trajectoryPlot == 'coloredlines':
                # Construct segments
                points = np.array([varX, varY]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                ##### Plot collection
                lc = LineCollection(segments, cmap=plt.get_cmap('hsv'), norm=plt.Normalize(0, timeFromStart))
                lc.set_array(np.array(timeFromStart))
                lc.set_linewidth(1)
                
                ax = axSP.add_collection(lc)
            if trajectoryPlot == 'scatter':
                plt.scatter(varX, varY)
            if trajectoryPlot == 'lines':
                plt.plot(varX, varY)
                
            ############# Plot sections
            if trajectoryPlot == 'sections':
                idxLabs = np.where(np.logical_and(varNamesAll != varNamesAll[col],varNamesAll != varNamesAll[row]))
                
                # The section is defined by fixing an interval on the third variable
                idxVar3 = idxLabs[0][0]
                idxVar4 = idxLabs[0][1]

                labVar3 = varLabels[idxVar3]
                labVar4 = varLabels[idxVar4]
                
                minInterval = sectionIntervals[idxVar3,0]
                maxInterval = sectionIntervals[idxVar3,1]
                
                print('Sect. for ', labVar3, ': ', minInterval, '-', maxInterval)
                idxData = np.where(np.logical_and(varData[:,idxVar3] >= minInterval, varData[:,idxVar3] <= maxInterval))
                varXsect = varX[idxData[0]]
                varYsect = varY[idxData[0]]
                
                # The colors of the dots are defined by the fourth variable
                if (labVar4 == 'IMF' or labVar4 == 'MM' or labVar4 == 'WAR') and boolLogPlot:
                    if (labVar4 == 'IMF') or (labVar4 == 'MM'):
                        offset = 0.005
                    elif labVar4 == 'WAR':
                        offset = 0.01
                    else:
                        offset = 0.0
                    var4col = 10*np.log10(varData[idxData[0],idxVar4] + offset)
                    labVar4 = labVar4 + ' [dB]'
                else:
                    var4col = varData[idxData[0],idxVar4]

                # Scatter
                vmin = axesLimits[idxVar4,0]
                vmax = axesLimits[idxVar4,1]
                vmin = np.percentile(var4col,5)
                vmax = np.percentile(var4col,95)
                
                scIm = plt.scatter(varXsect, varYsect, c=var4col, vmin=vmin, vmax=vmax, s=1.5, edgecolor='none')
                cbar = plt.colorbar(scIm)
                cbar.set_label(labVar4, labelpad=-15, y=1.10, rotation=0, fontsize=9)
                #cbar.ax.set_title(labVar4)

                #titleStrSP = 'Sect. for ' + labVar3 + ': ' + str(minInterval) + '-' + str(maxInterval)
                titleStrSP = 'Surface section for ' + str(int((maxPercSec+minPercSec)/2))+ '-pctile \n' + labVar3 + ' in ' + str(fmt2 % minInterval) + '-' + str(fmt2 % maxInterval)
                
            # Axis limits and title
            plt.xlim(axesLimits[col,0],axesLimits[col,1])
            plt.ylim(axesLimits[row,0],axesLimits[row,1])
            plt.title(titleStrSP, fontsize=9)
        
        ############# Plot 2d histogram or kernel density
        criterion_NxN = ((row < col) and (nrRows == nrCols))
        criterion_NxP = ((nrRows != nrCols) and ((col != 0) or (row != 0)))

        if criterion_NxN or criterion_NxP:
            print('Drawing ', densityPlot,' at row, col=', row, ',',col, ' - ', varLabels[row], ' vs ', varLabels[col])
            # Compute correlation
            beta, intercept, r_beta, p_value, std_err = stats.linregress(varX, varY)
            if densityPlot == 'kde':
                # Compute kernel density
                X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                values = np.vstack([varX, varY])
                kernel = stats.gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X.shape)
                
                # Plot kernel density
                #plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], aspect='auto')
                classLimits = np.concatenate((np.arange(0.001,0.01,0.001), np.arange(0.01,0.06,0.01),np.arange(0.1,1.1,0.1)))
                normLog = colors.LogNorm(vmin=0, vmax=5)
                Zmax = np.max(np.max(Z))
                histIm = plt.contourf(X, Y, Z/Zmax, classLimits, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], aspect='auto')
                #plt.title('KDE',fontsize=10)
            
            if densityPlot == '2dhist':
                # Plot 2D histogram
                colorList = ['darkblue', 'green', 'yellow', 'orange', 'red', 'blueviolet', 'lavender']
                cmapHist = colors.LinearSegmentedColormap.from_list('cmapHist', colorList, N=256)
                
                # X-bins
                #xbins = np.linspace(xmin, xmax, nrBinsX)
                #ybins = np.linspace(ymin, ymax, nrBinsY)
                xbins = np.linspace(axesLimits[col,0], axesLimits[col,1], nrBinsX)
                ybins = np.linspace(axesLimits[row,0], axesLimits[row,1], nrBinsY)
                
                # Compute histogram
                counts, _, _ = np.histogram2d(varX, varY, bins=(xbins, ybins))
                nrSamples = len(varX)
                counts[counts == 0] = np.nan
                counts = counts/nrSamples*100
                countsMask = ma.array(counts,mask=np.isnan(counts))
                
                # Draw histogram
                maxFreq = nrBinsX/(nrBinsX/0.3)
                histIm = plt.pcolormesh(xbins, ybins, countsMask.T, cmap=cmapHist, vmax = maxFreq)
                #cbar = plt.colorbar(histIm)
                
                # if varLabels[row] == 'IMF' or varLabels[row] == 'WAR':
                    # axSP.set_yscale('log')
                    
                # Directly plot histogram
                #plt.hist2d(varX, varY, bins=20, cmin=1, cmap=cmapHist) #, norm=LogNorm()) # gist_ncar, jet, spectral
                
                axSP.set_xlim(axesLimits[col,0],axesLimits[col,1])
                axSP.set_ylim(axesLimits[row,0],axesLimits[row,1])
                
                #axSP.set_aspect(1./ax.get_data_ratio())

                corrText = 'R=' + str(fmt2 % r_beta)
                if np.abs(r_beta) > 0.3:
                    colore = 'red'
                else:
                    colore = 'black'
                axSb = plt.gca()
                plt.text(0.74, 0.93, corrText, transform=axSb.transAxes, fontsize=ftSize, color=colore,bbox=dict(facecolor='white', edgecolor='black', pad=1.0))
            
        # Plot time series on the diagonal
        criterion_NxN = (row == col) and (len(varX) <= 288*1) and (nrRows == nrCols)
        
        if criterion_NxN: # plot max 5 days
            axDiag=plt.gca()
            plt.tick_params(bottom = 'off')
            plt.xticks(rotation=90)
            axDiag.plot(timeStampsDt, varY, 'b-')
            xfmt = md.DateFormatter('%H') #'%Y-%m-%d %H:%M:%S'
            axDiag.xaxis.set_major_formatter(xfmt)
            
        # Plot 1d histogram
        criterion_NxN = (row == col) and (len(varX) > 288*1) and (nrRows == nrCols)
        criterion_NxP = ((nrRows != nrCols) and row == 0 and col == 0)
        
        if criterion_NxN or criterion_NxP:
            # Compute 1d histogram
            counts, bins = np.histogram(varY, bins=nrBinsY, range=axesLimits[row,:])
            nrSamples = len(varY)
            counts = 100.0*counts/float(nrSamples)
            medianVal = np.nanmedian(varY)
            meanVal = np.nanmean(varY)
            stdVal = np.nanstd(varY)
            width = 0.4 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2.0
            
            # Plot hist
            axSb = plt.gca()
            plt.bar(center, counts, align='center', width=width, color='blue', edgecolor='blue')
            textMedian = r'median = ' + str("%0.2f" % medianVal)
            textMean = r'$\mu$ = ' + str("%0.2f" % meanVal)
            textStd = r'$\sigma$ = ' + str("%0.2f" % stdVal)
            plt.text(0.05, 0.90, textMedian, transform=axSb.transAxes, fontsize=ftSize)
            plt.text(0.05, 0.84, textMean, transform=axSb.transAxes, fontsize=ftSize)
            plt.text(0.05, 0.78, textStd, transform=axSb.transAxes, fontsize=ftSize)
            maxPerc = 15
            plt.ylim([0, maxPerc])
        # Axis labels
        if col == 0 and (nrRows == nrCols):
            plt.ylabel(varYLab, fontsize=12)
        if (row == nrRows-1) and (nrRows == nrCols):
            plt.xlabel(varXLab, fontsize=12)
        if row == 0 and (nrRows == nrCols):
            plt.title(varXLab, fontsize=12)
        if (col == nrCols-1) and (nrRows == nrCols):
            axSP.yaxis.set_label_position("right")
            plt.ylabel(varYLab, fontsize=12)
        # Axis labels for non square subplot matrix
        if (nrRows != nrCols):
            plt.xlabel(varXLab, fontsize=12)
            plt.ylabel(varYLab, fontsize=12)
            if row == 0 and col == 0:
                plt.ylabel('Frequency [%]', fontsize=12)

fig.tight_layout()
fig.subplots_adjust(top=0.92, right=0.8)

# Main title
titleStr = product + ': ' + str(timeStampsDt[0]) + ' - ' + str(timeStampsDt[len(timeStampsDt)-1])
plt.suptitle(titleStr, fontsize=16)

# Colorbar for time from start of event
if trajectoryPlot == 'coloredlines':
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(ax, cax = cbar_ax) # bug with fraction=0.03
    if daysFromStart > 5:
        cbar.ax.set_ylabel('Days from start of event')
    else:
        cbar.ax.set_ylabel('Hours from start of event')

# Write text on data conditions for analysis and plotting        
if trajectoryPlot == 'sections':
    xoffset = 0.84
    yoffset = 0.92
    lineSpacing = 0.03
    fig.text(xoffset, yoffset, "Conditions:", fontsize=12, color='blue')
    
    if (args.minR < 0.01): 
        textConditions = r"Rmin = " + (fmt3 % args.minR) + r' mm/hr'
    else:
        textConditions = r"Rmin = " + (fmt2 % args.minR) + r' mm/hr'
    fig.text(xoffset, yoffset-1*lineSpacing, textConditions, fontsize=12, color='blue')    
    
    if args.wols  == 0:
        textConditions = "OLS"
    if args.wols == 1:
        textConditions = "Weighted OLS"
    fig.text(xoffset, yoffset-2*lineSpacing, textConditions, fontsize=12, color='blue')
    
    textConditions = "WAR $\geq $ " + str(fmt1 % warThreshold) + " %"
    fig.text(xoffset, yoffset-3*lineSpacing, textConditions, fontsize=12, color='blue')
    
    textConditions = r"$|r_\beta|$  $\geq $ " + str(fmt2 % betaCorrThreshold)
    fig.text(xoffset, yoffset-5*lineSpacing, textConditions, fontsize=12, color='blue')
    textConditions = r"$\frac{N_\beta}{N_{WAR}}$ = $\frac{" + str(nrSamplesBetasWAR) + r"}{" + str(nrSamplesWAR) + r"}$" #+ str(nrSamplesBetasWAR) str(nrSamplesWAR)
    fig.text(xoffset, yoffset-6*lineSpacing, textConditions, fontsize=12, color='blue')
    textConditions = r"$\frac{N_\beta}{N_{WAR}}$ = " + str(fmt1 % fractionValidBetas) + " %"
    fig.text(xoffset, yoffset-7.2*lineSpacing, textConditions, fontsize=12, color='blue')
    
    # Variables acronyms
    fig.text(xoffset, 0.07, "WAR = Wet area ratio", fontsize=11, color='black')
    fig.text(xoffset, 0.05, "MM = Marginal mean", fontsize=11, color='black')
    fig.text(xoffset, 0.02, "dB = decibel", fontsize=11, color='black')
    
###### Save figure 
fileName = outBaseDir + product + timeStartStr + '-' + timeEndStr +  '0_' + \
'Rgt' + str(args.minR) + '_WOLS' + str(args.wols) + '_00005_attractorSubplots_warGt' + str("%0.1f" % warThreshold) + '_' + timeAccumMinStr + '.png'
print('Saving: ',fileName)
plt.savefig(fileName, dpi=300)


        
