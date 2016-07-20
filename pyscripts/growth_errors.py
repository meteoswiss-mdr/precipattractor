#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

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
from scipy.spatial.distance import pdist,cdist
import datetime
import time
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
inBaseDir = '/scratch/lforesti/data/' # '/store/msrad/radar/precip_attractor/data/' #'/scratch/lforesti/data/'
outBaseDir = '/users/lforesti/results/'
pltType = 'spread' #'evolution' 'spread'

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Plot radar rainfall field statistics.')
parser.add_argument('-start', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-wols', default=0, type=int,help='Whether to use the weighted ordinary leas squares or not in the fitting of the power spectrum.')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
parser.add_argument('-minWAR', default=1, type=float,help='Minimum WAR threshold for plotting.')
parser.add_argument('-minCorrBeta', default=0.5, type=float,help='Minimum correlation coeff. for beta for plotting.')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-temp', default=5, type=int,help='Temporal sampling of the products [minutes].')
parser.add_argument('-format', default='netcdf', type=str,help='Format of the file containing the statistics [csv,netcdf].')
parser.add_argument('-plt', default='spread', type=str,help='Plot type [spread, evolution].')

args = parser.parse_args()

product = args.product
pltType = args.plt
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

############### OPEN FILES WITH STATS
if args.format == 'csv':
    arrayStats, variableNames = io.csv_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
    product = product, timeAccumMin = timeSampMin, minR=args.minR, wols=args.wols)
elif args.format == 'netcdf':
    arrayStats, variableNames = io.netcdf_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
    product = product, timeAccumMin = timeAccumMin, minR=args.minR, wols=args.wols)
else:
    print('Please provide a valid file format.')
    sys.exit(1)

# Generate list of datetime objects and absolute time
timeIntList = dt.get_column_list(arrayStats, 0)
timeStamps_datetime = ti.timestring_array2datetime_array(timeIntList)

timeStamps_absolute = []
for t in range(0,len(timeStamps_datetime)):
    absTime = ti.datetime2absolutetime(timeStamps_datetime[t])
    timeStamps_absolute.append(absTime)
    
# Convert list of lists to numpy arrays
arrayStats = np.array(arrayStats)
timeStamps_datetime = np.array(timeStamps_datetime)
timeStamps_absolute = np.array(timeStamps_absolute)

# Check if there are data
print(arrayStats.shape,' samples found.')
print('Variables from file: ', variableNames)

if (len(arrayStats) == 0) & (args.format == 'csv'):
    print("No data found in CSV files.")
    sys.exit(1)
if (len(arrayStats) == 0) & (args.format == 'netcdf'):
    print("No data found in NETCDF files.")
    sys.exit(1)

#################################################################################
####################### ANALYSE GROWTH OF ERRORS
varNames = ['war', 'r_mean', 'beta1', 'beta2']
warThreshold = args.minWAR
betaCorrThreshold = args.minCorrBeta
useTrajectWholeArchive = True
independenceTimeHours = 24
logIMFWAR = False

logTime = True # Keep it True
logSpread = True # Keep it True
ylims = [10**-1.7,10**0.5]
maxLeadTimeMin = 60*24

# Definition of initial conditions
initialCondIntervals = [0.5, 0.01, 0.05, 0.05]
initialCondRange = [np.arange(5.0,55.0,5.0).tolist(),\
np.arange(0.2,3.0,0.2).tolist(),\
np.arange(1.5,2.5,0.1).tolist(),\
np.arange(2.5,3.5,0.1).tolist()]

minNrTraj = 20
print('Variables for plotting: ', varNames)

####################################################################################
####################### PREPARE DATA ###############################################
# Generate labels for plotting
varLabels = []
for var in range(0, len(varNames)):
    if varNames[var] == 'war':
        if logIMFWAR:
            varLabels.append('WAR [dB]')
        else:
            varLabels.append('WAR')
    if varNames[var] == 'r_mean':
        if logIMFWAR:
            varLabels.append('IMF [dB]')
        else:
            varLabels.append('IMF')
    if varNames[var] == 'beta1':
        varLabels.append(r'$\beta_1$')
    if varNames[var] == 'beta2':
        varLabels.append(r'$\beta_2$')

# Get indices of variables
indicesVars = dt.get_variable_indices(varNames, variableNames)
# Put indices into dictionary
dictIdx = dict(zip(varNames, indicesVars))
dictLabels = dict(zip(varNames, varLabels))

# WAR threshold
boolWAR = (arrayStats[:,dictIdx['war']] >= warThreshold) 
# Beta correlation threshold
boolBetaCorr = (arrayStats[:,dictIdx['beta1']+1] <= -betaCorrThreshold) & (arrayStats[:,dictIdx['beta2']+1] <= -betaCorrThreshold)
# Combination of thresholds
boolTot = boolWAR & boolBetaCorr

############### Select subset of data
arrayStats_subset = []
arrayStats_all = []
for var in range(0, len(varNames)):
    varName = varNames[var]
    if (varName == 'beta1') | (varName == 'beta2'):
        arrayStats_subset.append(-arrayStats[boolTot,dictIdx[varName]])
        arrayStats_all.append(-arrayStats[:,dictIdx[varName]])
    elif (varName == 'war') | (varName == 'r_mean'):
        if logIMFWAR == True:
            arrayStats_subset.append(dt.to_dB(arrayStats[boolTot,dictIdx[varName]]))
            arrayStats_all.append(dt.to_dB(arrayStats[:,dictIdx[varName]]))
        else:
            arrayStats_subset.append(arrayStats[boolTot,dictIdx[varName]])
            arrayStats_all.append(arrayStats[:,dictIdx[varName]])
            
    else:
        arrayStats_subset.append(arrayStats[boolTot,dictIdx[varName]])
        arrayStats_all.append(arrayStats[:,dictIdx[varName]])

# Convert lists to numpy arrays    
arrayStats_subset = np.array(arrayStats_subset).T
arrayStats_all = np.array(arrayStats_all).T
timeStamps_datetime_subset = np.array(timeStamps_datetime)[boolTot]
timeStamps_absolute_subset = timeStamps_absolute[boolTot]

# Replace bad values with NaNs        
#arrayStats_all[boolTot,:] = np.nan

# Compute summary stats for normalization and quantiles for automatic selection of initial conditions
if useTrajectWholeArchive == False:
    arrayStats_Mean = np.mean(arrayStats_subset, axis=0)
    arrayStats_Std = np.std(arrayStats_subset, axis=0)

else:
    arrayStats_Mean = np.mean(arrayStats_all, axis=0)
    arrayStats_Std = np.std(arrayStats_all, axis=0)

print('Means: ', arrayStats_Mean)
print('St.devs: ', arrayStats_Std)

# Set new initial conditions
arrayStats_minPerc = np.percentile(arrayStats_subset, 5, axis=0)
arrayStats_maxPerc = np.percentile(arrayStats_subset, 98, axis=0)
print('MinPerc: ', arrayStats_minPerc)
print('MaxPerc: ', arrayStats_maxPerc)

initialCondIntervals = (arrayStats_maxPerc - arrayStats_minPerc)/100.0

nrIntervals = 5
for var in range(0, len(varNames)):
    initialCondRange[var] = np.linspace(arrayStats_minPerc[var], arrayStats_maxPerc[var], nrIntervals).tolist()

print('Initial conditions: ', initialCondRange)
print('Initial intervals: ', initialCondIntervals)

####################################################################################################
############### SELECT INTERVAL FOR INITIAL CONDITIONS
nrLeadTimes = int(maxLeadTimeMin/timeSampMin)
nrDimensions = arrayStats_subset.shape[1]

fig = plt.figure(figsize=(13, 13))
ax = fig.add_axes()
ax = fig.add_subplot(111)

colormap = plt.cm.gist_rainbow
nrRowsColsSubplots = 2
p = 0

tic = time.clock()
for variable in range(0, len(varNames)): ## LOOP OVER VARIABLES
    analysisSteps = initialCondRange[variable]
    nrSteps = len(analysisSteps)
    p = p + 1 # subplot number
    
    print('\n')
    varMax = 0
    varMin = 999
    
    axSP = plt.subplot(nrRowsColsSubplots, nrRowsColsSubplots, p)
    print('Subplot nr: ', p)
    
    for step in range(0, nrSteps): ## LOOP OVER STEPS FOR INITIAL CONDITIONS
        # Define min and max value for initial conditions
        minInit = analysisSteps[step]
        maxInit = analysisSteps[step] + initialCondIntervals[variable]
        
        # Select data and time stamps of initial conditions
        initialConditions_data = (arrayStats_subset[:,variable] >= minInit) & (arrayStats_subset[:,variable] <= maxInit)
        initialConditions_timestamps = timeStamps_absolute_subset[initialConditions_data]
        
        nrInitPoints = np.sum(initialConditions_data == True)
        print(nrInitPoints, ' samples in ', varLabels[variable], ' range ', minInit,'-',maxInit)
        
        # Compute time differences between consecutive time stamps of initial conditions
        timeDiffs = np.diff(initialConditions_timestamps)
        
        # Create array of time stamps that have a certain temporal independence (e.g. 24 hours)
        independenceTimeSecs = 60*60*independenceTimeHours
        timeDiffsAccum = 0
        initialConditions_timestamps_indep = []
        for i in range(0,nrInitPoints-1):
            if (timeDiffs[i] >= independenceTimeSecs) | (timeDiffsAccum >= independenceTimeSecs):
                initialConditions_timestamps_indep.append(initialConditions_timestamps[i])
                timeDiffsAccum = 0
            else:
                # Increment the accumulated time difference to avoid excluding the next sample 
                # if closer than 24 hours from the previous (if not included) but further than 24 hours than the before the previous
                timeDiffsAccum = timeDiffsAccum + timeDiffs[i]
                
        initialConditions_timestamps_indep = np.array(initialConditions_timestamps_indep)
        
        nrInitPoints = len(initialConditions_timestamps_indep)
        print(nrInitPoints, ' independent samples in ', varLabels[variable], ' range ', minInit,'-',maxInit)
        
        ################## GET ANALOGUE DATA SEQUENCES FOLLOWING TIME STAMPS
        # Loop over such points and get data sequences
        trajectories = [] # list of trajectories
        for i in range(0,nrInitPoints):
            tsAbs = initialConditions_timestamps_indep[i]
            if useTrajectWholeArchive == False:
                idx = np.where(timeStamps_absolute_subset == tsAbs)[0]
            else:
                idx = np.where(timeStamps_absolute == tsAbs)[0]
            if len(idx) != 1:
                print(idx)
                print(timeStamps_absolute[idx[0]], timeStamps_absolute[idx[1]])
                print('You have duplicate time stamps in your dataset. Taking the first...')
                sys.exit(1)

            indicesSequence = np.arange(idx,idx+nrLeadTimes)

            # Handle sequences that go beyond the dataset limits
            if np.sum(indicesSequence >= len(timeStamps_absolute_subset)) > 0:
                indicesSequence = indicesSequence[indicesSequence < len(timeStamps_absolute_subset)]
                
            # Select data sequences    
            if useTrajectWholeArchive == False:
                sequenceTimes = timeStamps_absolute_subset[indicesSequence]
                sequenceData = arrayStats_subset[indicesSequence,:]
            else:
                sequenceTimes = timeStamps_absolute[indicesSequence]
                sequenceData = arrayStats_all[indicesSequence]

            # Check times and radar operation and replace with NaNs
            timeDiffMin = np.array(np.diff(sequenceTimes)/60, dtype=int)
            
            # Collect trajectories
            nrInvalidSamples = np.sum(timeDiffMin != timeSampMin)
            if (nrInvalidSamples == 0) & (len(sequenceTimes) == nrLeadTimes):
                trajectories.append(sequenceData)

        trajectories = np.array(trajectories)
        
        if len(trajectories) > minNrTraj:
            #print(trajectories.shape[0], ' x ', trajectories.shape[1], ' x ', trajectories.shape[2], '($N_{analogue}$) x ($N_{leadtimes}$) x ($N_{dim}$)')

            ################## COMPUTE SPREAD OF TRAJECTORIES
            spreadArray = []
            for lt in range(0,nrLeadTimes):
                dataLeadTime = trajectories[:,lt,:]
                spreadLeadTime = np.std(dataLeadTime/arrayStats_Std, axis=0)
                spreadArray.append(spreadLeadTime)

            spreadArray = np.array(spreadArray)

            ################## PLOTTING ################################################################################
            linewidth=2.0
            labelFontSize = 16
            legendFontSize = 12
            axesTicksFontSize = 14
            
            plt.tick_params(axis='both', which='major', labelsize=axesTicksFontSize)
            # Generate lead times
            leadTimesMin = []
            for lt in range(0,nrLeadTimes):
                leadTimesMin.append(lt*timeSampMin)
            leadTimesMin = np.array(leadTimesMin)
               
            # Plot growth of spread
            legTxt = ' Range ' + str(fmt2 % minInit) + '-' + str(fmt2 % maxInit) + ' (N = ' + str(trajectories.shape[0]) + ')'
            
            if pltType == 'spread':
                if (logTime == True) & (logSpread == True):
                    axSP.loglog(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                elif logTime == True:
                    axSP.semilogx(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                elif logSpread == True:
                    axSP.semilogy(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                else:
                    axSP.plot(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
            
            # Plot evolution of trajectories
            stepEvol = 3
            if (pltType == 'evolution') & (step == stepEvol):
                if (logTime == True) & (logSpread == True):
                    #axSP.loglog(leadTimesMin/60,trajectories[1:20,:,variable].T, color='blue')
                    axSP.plot(leadTimesMin/60,trajectories[1:20,:,variable].T, color='blue')
                if (logTime == True) & (logSpread == False):
                    axSP.semilogx(leadTimesMin/60,trajectories[1:20,:,variable].T, color='blue')
                else:
                    axSP.plot(leadTimesMin/60,trajectories[1:20,:,variable].T, color='blue')
                
            # Plot spread of trajectories
            if pltType == 'spread':
                # Line colors
                colors = [colormap(i) for i in np.linspace(0, 1,len(axSP.lines))]
                for i,j in enumerate(axSP.lines):
                    j.set_color(colors[i])
                
                # Add legend
                plt.ylim(ylims)
                if (logTime == True) & (logSpread == True):
                    plt.xlim([timeSampMin/60, maxLeadTimeMin/60])
                    axSP.legend(loc='lower right', fontsize=legendFontSize)
                elif logTime == True:
                    axSP.legend(loc='upper left', fontsize=legendFontSize)
                elif logSpread == True:
                    axSP.legend(loc='lower right', fontsize=legendFontSize)
                else:
                    axSP.legend(loc='lower right', fontsize=legendFontSize)
            
            # Add labels and title
            plt.xlabel('Lead time [hours]', fontsize=labelFontSize)
            
            if (pltType == 'evolution') & (step == stepEvol):
                plt.ylabel(varLabels[variable], fontsize=labelFontSize)
                strTitle = 'Evolution of ' + varLabels[variable] + ' starting at ' + str(fmt2 % minInit) + '-' + str(fmt2 % maxInit)
                plt.title(strTitle, fontsize=18)
            if pltType == 'spread':
                plt.ylabel('Norm. st. deviation', fontsize=labelFontSize)
                strTitle = 'Spread growth for ' + varLabels[variable]
                plt.title(strTitle, fontsize=18)

toc = time.clock()
print('Total elapsed time: ', toc-tic, ' seconds.')

# Main title
titleStr = 'Growth of spread in the attractor for \n' + product + ': ' + str(timeStamps_datetime[0]) + ' - ' + str(timeStamps_datetime[len(timeStamps_datetime)-1])
plt.suptitle(titleStr, fontsize=20)

# Save figure
fileName = outBaseDir + product + '_' + pltType + '_' + timeStartStr + '-' + timeEndStr +  '0_' + \
'Rgt' + str(args.minR) + '_WOLS' + str(args.wols) + '_00005_warGt' + str("%0.1f" % warThreshold) + '_logIMFWAR' + str(int(logIMFWAR)) + '_' + timeAccumMinStr + '.png'
print('Saving: ',fileName)
plt.savefig(fileName, dpi=300)