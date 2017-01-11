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
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as md
from matplotlib.collections import LineCollection
import pylab
mpl.rc('text.latex', preamble='\usepackage{color}')

from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.optimize import curve_fit
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
            
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
import stat_tools_attractor as st

fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"
np.set_printoptions(precision=4)
    
################# DEFAULT ARGS #########################
inBaseDir = '/scratch/lforesti/data/' # '/store/msrad/radar/precip_attractor/data/' #'/scratch/lforesti/data/'
outBaseDir = '/users/lforesti/results/'
tmpBaseDir = '/scratch/lforesti/tmp/'
pltType = 'spread' #'evolution' 'spread'
timeSampMin = 5
spreadMeasure = 'scatter'#'std' or 'scatter'

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
parser.add_argument('-refresh', default=0, type=int,help='Whether to refresh the binary .npy archive or not.')

args = parser.parse_args()

refreshArchive = bool(args.refresh)
print('Refresh archive:', refreshArchive)
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

if spreadMeasure != 'std' and spreadMeasure != 'scatter':
    print('The measure of spread should be either std or scatter')
    sys.exit(1)
if spreadMeasure == 'std':
    txtYlabel = 'Normalized st. deviation'
if spreadMeasure == 'scatter':
    txtYlabel = 'Normalized half scatter'
    
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
    ## Open whole list of CSV or netCDF files
    if args.format == 'csv':
        arrayStats, variableNames = io.csv_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
        product = product, timeAccumMin = timeSampMin, minR=args.minR, wols=args.wols)
    elif args.format == 'netcdf':
        arrayStats, variableNames = io.netcdf_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', \
        product = product, timeAccumMin = timeAccumMin, minR=args.minR, wols=args.wols, variableBreak = 0)
    else:
        print('Please provide a valid file format.')
        sys.exit(1)
        
    # Check if there are data
    if (len(arrayStats) == 0) & (args.format == 'csv'):
        print("No data found in CSV files.")
        sys.exit(1)
    if (len(arrayStats) == 0) & (args.format == 'netcdf'):
        print("No data found in NETCDF files.")
        sys.exit(1)

## Save data into a single binary bython file to speed up further analysis with same dataset
arrayData = []
if refreshArchive == True:
    np.save(tmpArchiveFileName, arrayStats)
    np.save(tmpArchiveFileNameVariables, variableNames)
    print('Saved:',tmpArchiveFileName)
    
################ Fill both datetime and data arrays with NaNs where there is no data
# Generate list of datetime objects
timeIntList = dt.get_column_list(arrayStats, 0)
timeStamps_datetime = ti.timestring_array2datetime_array(timeIntList)

nrSamples = len(timeStamps_datetime)
print('Number of analysed radar fields in archive: ', nrSamples)
nrSamplesTotal = int((timeStamps_datetime[nrSamples-1] - timeStamps_datetime[0]).total_seconds()/(timeSampMin*60))
print('Number of missing fields: ', nrSamplesTotal-nrSamples)

# Fill attractor array with NaNs to consider every missing time stamp
arrayStats, timeStamps_datetime = dt.fill_attractor_array_nan(arrayStats, timeStamps_datetime)

print(len(arrayStats), len(timeStamps_datetime), 'samples after filling holes with NaNs.')
print('Variables from file: ', variableNames)

######## Prepare numpy arrays
timeStamps_absolute = ti.datetime2absolutetime(np.array(timeStamps_datetime))

# Convert list of lists to numpy arrays
arrayStats = np.array(arrayStats)
timeStamps_datetime = np.array(timeStamps_datetime)
timeStamps_absolute = np.array(timeStamps_absolute)

#################################################################################
####################### PARAMETERS TO ANALYZE GROWTH OF ERRORS
varNames = ['war', 'r_cmean', 'r_mean', 'eccentricity', 'beta1', 'beta2']

logIMFWAR = True
logTime = True # Keep it True (or false to check exponential growth of errors)
logSpread = True # Keep it True

maxLeadTimeHours = 96
ylims = [10**-1.7,10**0.5]

# Selection criteria for valid trajectories
warThreshold = args.minWAR
betaCorrThreshold = args.minCorrBeta
independenceTimeHours = 1
minNrTraj = 20 # Minimum number of trajectories
nrIQR = 5 # Multiplier of the IQR to define a sample as outlier
verbosity = 1

# Whether to plot the function fits to the growth of errors one by one
plotFits = False

print('Variables for plotting: ', varNames)

####################################################################################
####################### PREPARE DATA ###############################################
maxLeadTimeMin = 60*maxLeadTimeHours

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
    if varNames[var] == 'r_cmean':
        if logIMFWAR:
            varLabels.append('MM [dB]')
        else:
            varLabels.append('MM')
    if varNames[var] == 'eccentricity':
            if logIMFWAR:
                varLabels.append('1-eccentricity [dB]')
            else:
                varLabels.append('Eccentricity')
    if varNames[var] == 'beta1':
        varLabels.append(r'$\beta_1$')
    if varNames[var] == 'beta2':
        varLabels.append(r'$\beta_2$')

# Get indices of variables
indicesVars = dt.get_variable_indices(varNames, variableNames)
# Put indices into dictionary
dictIdx = dict(zip(varNames, indicesVars))
dictLabels = dict(zip(varNames, varLabels))
print(dictIdx)

# WAR threshold
boolWAR = (arrayStats[:,dictIdx['war']] >= warThreshold) 
# Beta correlation threshold
boolBetaCorr = (np.abs(arrayStats[:,dictIdx['beta1']+1]) >= np.abs(betaCorrThreshold)) & (np.abs(arrayStats[:,dictIdx['beta2']+1]) >= np.abs(betaCorrThreshold))
# Combination of thresholds
boolTot = np.logical_and(boolWAR == True, boolBetaCorr == True)

############### Select subset of variables and change sign of beta
arrayStats_attractor = []
for var in range(0, len(varNames)):
    varName = varNames[var]
    if (varName == 'beta1') | (varName == 'beta2'):
        arrayStats_attractor.append(-arrayStats[:,dictIdx[varName]])
    elif (varName == 'war') | (varName == 'r_mean') | (varName == 'r_cmean') | (varName == 'eccentricity'):
        if logIMFWAR == True:
            if varName == 'eccentricity':
                arrayStats_attractor.append(dt.to_dB(1-arrayStats[:,dictIdx[varName]]))
            else:
                arrayStats_attractor.append(dt.to_dB(arrayStats[:,dictIdx[varName]]))
        else:
            arrayStats_attractor.append(arrayStats[:,dictIdx[varName]])  
    else:
        arrayStats_attractor.append(arrayStats[:,dictIdx[varName]])

# Convert lists to numpy arrays    
arrayStats_attractor = np.array(arrayStats_attractor).T

# Replace "bad" samples with NaNs  
arrayStats_attractor[boolWAR==False,:] = np.nan

# Calculate global statistics on the data
arrayStats_Mean = np.nanmean(arrayStats_attractor, axis=0)
arrayStats_Std = np.nanstd(arrayStats_attractor, axis=0)
arrayStats_Scatter = st.nanscatter(arrayStats_attractor, axis=0)

## Compute data increments (changes from one time instant to the other)
arrayStats_increments = np.diff(arrayStats_attractor, axis=0)
# Set first increment equal to the second 
arrayStats_increments = np.vstack((arrayStats_increments[0,:], arrayStats_increments)) 

## Compute global statistics on the data increments
arrayStats_increments_Mean = np.nanmean(arrayStats_increments, axis=0)
arrayStats_increments_Std = np.nanstd(arrayStats_increments, axis=0)
Q25 = np.nanpercentile(arrayStats_increments,25, axis=0)
Q75 = np.nanpercentile(arrayStats_increments,75, axis=0)
arrayStats_increments_IQR =  Q75 - Q25

# Print info on statistics of data and increments
if verbosity >= 1:
    print('Means  : ', arrayStats_Mean)
    print('St.devs: ', arrayStats_Std)
    print('Scatter: ', arrayStats_Scatter)
    print('Means increments  : ', arrayStats_increments_Mean)
    print('St.devs increments: ', arrayStats_increments_Std)
    print('IQR increments    : ', arrayStats_increments_IQR)

##########PLOT INCREMENTS
# Plot time series of increments
plotIncrements = True
if plotIncrements:
    nrRowsSubplots = 2
    nrColsSubplots = 3
    p=1
    fig = plt.figure(figsize=(22,10))
    for var in range(0, len(varNames)):
        ax = plt.subplot(nrRowsSubplots, nrColsSubplots, p)
        plt.plot(arrayStats_increments[:,var])
        ax.axhline(y=Q25[var] - nrIQR*arrayStats_increments_IQR[var],color='r')
        ax.axhline(y=Q75[var] + nrIQR*arrayStats_increments_IQR[var],color='r')
        plt.title('Time series increments for ' + varNames[var])
        p += 1
    plt.show()

    # Plot histogram of increments
    p=1
    fig = plt.figure(figsize=(22,10))
    for var in range(0, len(varNames)):
        plt.subplot(nrRowsSubplots, nrColsSubplots, p)
        histRange = [Q25[var] - nrIQR*arrayStats_increments_IQR[var], Q75[var] + nrIQR*arrayStats_increments_IQR[var]]
        bins = np.hstack((np.nanmin(arrayStats_increments[:,var]), np.linspace(histRange[0],histRange[1], 50), np.nanmax(arrayStats_increments[:,var])))
        n, bins, patches = plt.hist(arrayStats_increments[:,var], 50, range=histRange, facecolor='green', alpha=0.75)
        plt.title('Histogram of increments for ' + varNames[var])
        p += 1
    plt.show()

# Calculate global statistics on the data by removing the bad increments
arrayStats_attractor_nanincrements = arrayStats_attractor.copy()
for var in range(0, len(varNames)):
    histRange = [Q25[var] - nrIQR*arrayStats_increments_IQR[var], Q75[var] + nrIQR*arrayStats_increments_IQR[var]]
    boolGoodIncrementsVar = (arrayStats_increments[:,var] >= histRange[0]) & (arrayStats_increments[:,var] <= histRange[1])
    arrayStats_attractor_nanincrements[~boolGoodIncrementsVar,var] = np.nan

arrayStats_Mean = np.nanmean(arrayStats_attractor_nanincrements, axis=0)
arrayStats_Std = np.nanstd(arrayStats_attractor_nanincrements, axis=0)
arrayStats_Scatter = st.nanscatter(arrayStats_attractor_nanincrements, axis=0)

# Print info on statistics of data (without bad increments)
if verbosity >= 1:
    print('Means  : ', arrayStats_Mean)
    print('St.devs: ', arrayStats_Std)
    print('Scatter: ', arrayStats_Scatter)

###########INITIAL CONDITIONS
##### Set the initial conditions of analogues intelligently (using percentiles)
arrayStats_minPerc = np.nanpercentile(arrayStats_attractor, 20, axis=0)
arrayStats_maxPerc = np.nanpercentile(arrayStats_attractor, 90, axis=0)
if verbosity >= 1:
    print('MinPerc: ', arrayStats_minPerc)
    print('MaxPerc: ', arrayStats_maxPerc)

initialCondIntervals = (arrayStats_maxPerc - arrayStats_minPerc)/100.0

nrIntervals = 5
initialCondRange = []
for var in range(0, len(varNames)):
    initialCondRange_variable = np.linspace(arrayStats_minPerc[var], arrayStats_maxPerc[var], nrIntervals).tolist()
    initialCondRange.append(initialCondRange_variable)
    
print('Initial conditions: ', np.array(initialCondRange))
print('Initial intervals: ', np.array(initialCondIntervals))

####################################################################################################
############### COMPUTE GROWTH OF ERRORS AND PLOT RESULTS
nrLeadTimes = int(maxLeadTimeMin/timeSampMin)
nrDimensions = arrayStats_attractor.shape[1]

# Generate lead times
leadTimesMin = []
for lt in range(0,nrLeadTimes):
    leadTimesMin.append(lt*timeSampMin)
leadTimesMin = np.array(leadTimesMin)

colormap = plt.cm.gist_rainbow # plt.cm.gray
nrRowsSubplots = 2
nrColsSubplots = 3
p = 0

if nrRowsSubplots == nrColsSubplots:
    fgSize = (13, 13)
else:
    fgSize = (20, 13)

fig = plt.figure(figsize=fgSize)    
ax = fig.add_axes()
ax = fig.add_subplot(111)

tic = time.clock()
for variable in range(0, len(varNames)): ## LOOP OVER VARIABLES
    analysisSteps = initialCondRange[variable]
    nrSteps = len(analysisSteps)
    p = p + 1 # subplot number
    
    print('\n')
    varMax = 0
    varMin = 999
    
    axSP = plt.subplot(nrRowsSubplots, nrColsSubplots, p)
    print('Subplot nr: ', p, ', variable: ', varNames[variable])
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    plot_lines = []
    decorr_time_hours = []
    for step in range(0, nrSteps): ## LOOP OVER STEPS FOR INITIAL CONDITIONS
        # Define min and max values for initial conditions
        minInit = analysisSteps[step]
        maxInit = analysisSteps[step] + initialCondIntervals[variable]
        
        if (varNames[variable] == 'war' or varNames[variable] == 'r_mean' or varNames[variable] == 'r_cmean' or varNames[variable] == 'eccentricity') and logIMFWAR == True:
            if varNames[variable] == 'eccentricity':
                minInitLab = dt.from_dB(minInit)
                maxInitLab = dt.from_dB(maxInit)

            else:
                minInitLab = dt.from_dB(minInit)
                maxInitLab = dt.from_dB(maxInit)
        else:
            minInitLab = minInit
            maxInitLab = maxInit
        
        # Select data and time stamps of initial conditions
        initialConditions_data = (arrayStats_attractor[:,variable] >= minInit) & (arrayStats_attractor[:,variable] <= maxInit)
        initialConditions_timestamps = timeStamps_absolute[initialConditions_data]
        
        nrInitPoints = np.sum(initialConditions_data == True)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        print(nrInitPoints, ' starting points in ', varLabels[variable], ' range ', minInit,'-',maxInit)
        
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
        print(nrInitPoints, ' independent starting points in ', varLabels[variable], ' range ', minInit,'-',maxInit)
        
        ################## GET ANALOGUE DATA SEQUENCES FOLLOWING TIME STAMPS
        # Loop over such points and get data sequences
        trajectories = [] # list of trajectories
        for i in range(0,nrInitPoints):
            tsAbs = initialConditions_timestamps_indep[i]
            idx = np.where(timeStamps_absolute == tsAbs)[0]
            if len(idx) != 1:
                print(idx)
                print(timeStamps_absolute[idx[0]], timeStamps_absolute[idx[1]])
                print('You have duplicate time stamps in your dataset. Taking the first...')
                sys.exit(1)

            indicesSequence = np.arange(idx,idx+nrLeadTimes)
            
            # Select data sequences    
            # Handle sequences that go beyond the dataset limits
            if np.sum(indicesSequence >= len(timeStamps_absolute)) > 0:
                indicesSequence = indicesSequence[indicesSequence < len(timeStamps_absolute)]
            
            sequenceTimes = timeStamps_absolute[indicesSequence]
            sequenceData = arrayStats_attractor[indicesSequence,:]
            increments = arrayStats_increments[indicesSequence,:]

            # Analyse increments of each time series and replace with NaNs if jumps are unrealistic
            minIncrements = Q25[variable] - nrIQR*arrayStats_increments_IQR[variable]
            maxIncrements = Q75[variable] + nrIQR*arrayStats_increments_IQR[variable]
            
            # Criterion to define whether an increment is unrealistically large
            #boolLargeIncrements  = np.abs(increments) >= arrayStats_Std[variable]
            boolLargeIncrements = (increments[:,variable] < minIncrements) | (increments[:,variable] > maxIncrements)
            boolLargeIncrements[0] = False # The increment of the first element from the one before the start of the sequence is not considered as wrong
            idxFirsBadIncrement = np.argmax(boolLargeIncrements == True)
            
            maxNrBadIncrements = 5
            if np.sum(boolLargeIncrements) > maxNrBadIncrements:
                # Replace all data with NaNs
                sequenceData[:,variable] = np.nan
            #else:
                # Replace data from first bad increment till the end with NaNs
                #sequenceData[idxFirsBadIncrement:,variable] = np.nan
            
            # Check the continuity of time stamps (no holes)
            timeDiffMin = np.array(np.diff(sequenceTimes)/60, dtype=int)
            # Nr of invalid samples not having the correct time stamp (it should be zero if correctly filled with NaNs)
            nrInvalidTimes = np.sum(timeDiffMin != timeSampMin)
            
            # Check how many valid data (not NaNs) you have in the future sequence
            nrValidSamples = np.sum(~np.isnan(sequenceData[:,variable]))
            nrConsecValidSamples = np.argmax(np.isnan(sequenceData[:,variable]))
            
            # Criteria to consider a sequence as valid
            minNrValidSamples = 36
            minNrConsecValidSamples = 12 # one hour from start
            
            # Collect valid trajectories
            criterion = (nrValidSamples >= minNrValidSamples) & (nrConsecValidSamples >= minNrConsecValidSamples) \
            & (nrInvalidTimes == 0) & (len(sequenceTimes) == nrLeadTimes)
            goodTrajectory = False
            if criterion == True:
                goodTrajectory = True
                trajectories.append(sequenceData)
            
            ### Print info on increments and valid samples...
            #print(increments[:,variable])
            if verbosity >= 2:
                print('Trajectory nr', i,'starting at', ti.absolutetime2datetime(tsAbs))
                print('Nr. invalid increments :', np.sum(boolLargeIncrements, axis=0))
                print('Valid increment limits:' , minIncrements, maxIncrements)
                print('First bad increment at index', idxFirsBadIncrement, 'with value', increments[idxFirsBadIncrement, variable])
                print('Nr. valid samples in sequence            : ', nrValidSamples, '/',nrLeadTimes)
                print('Nr. consecutive valid samples in sequence: ', nrConsecValidSamples, '/',nrLeadTimes)
                print('Valid trajectory?', goodTrajectory)
                print('---------------')
        # Append trajectory to the list of trajectories
        trajectories = np.array(trajectories)
        
        print(len(trajectories), ' valid trajectories in ', varLabels[variable], ' range ', minInit,'-',maxInit)
        if len(trajectories) > minNrTraj:
            #print(trajectories.shape[0], ' x ', trajectories.shape[1], ' x ', trajectories.shape[2], '($N_{analogue}$) x ($N_{leadtimes}$) x ($N_{dim}$)')

            ################## COMPUTE SPREAD OF TRAJECTORIES
            spreadArray = []
            for lt in range(0,nrLeadTimes):
                dataLeadTime = trajectories[:,lt,:]
                # Evaluate number of valid data
                nrValidPoints = np.sum(~np.isnan(dataLeadTime), axis=0)
                boolNrPoints = nrValidPoints < 20
                # Compute ensemble spread
                if spreadMeasure == 'std':
                    spreadLeadTime = np.nanstd(dataLeadTime/arrayStats_Std, axis=0)
                if spreadMeasure == 'scatter':
                    spreadLeadTime = st.nanscatter(dataLeadTime/(arrayStats_Scatter/2.0), axis=0)/2.0
                # Replace spread with nan if not enough samples for a given lead time
                if np.sum(boolNrPoints) >=1:
                    spreadLeadTime[boolNrPoints] = np.nan
                # Append spread
                spreadArray.append(spreadLeadTime)
            spreadArray = np.array(spreadArray)
            
            ################## DECORRELATION TIME ESTIMATION
            
            #### TESTS WITH DIFFERENT FITTED MODELS
            dB_shift_hr = 0.5
            if logTime:
                predictor = dt.to_dB(leadTimesMin/60 + dB_shift_hr)
            else:
                predictor = leadTimesMin/60
                
            predictand = dt.to_dB(spreadArray[:,variable])
            
            # Remove NaNs
            nans = np.isnan(predictand)
            predictor = predictor[~nans]
            predictand = predictand[~nans]
            
            if varNames[variable] == 'eccentricity':
                predictor = predictor[~np.isinf(predictand)]
                predictand = predictand[~np.isinf(predictand)]
                
            # Prediction grid
            predictor_grid = np.linspace(np.min(predictor), np.max(predictor), 1000)
            
            #### KERNEL RIDGE REGRESSION
            alphaVec = [0.1, 0.01]
            sigmaVec = np.arange(5.0, 5.5, 0.5)
            
            if len(alphaVec) > 1 or len(sigmaVec) > 1:
                # Grid search of parameters
                param_grid = {"alpha": alphaVec, "kernel": [RBF(length_scale) for length_scale in sigmaVec]}
                kr = KernelRidge()
                kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
            else:
                # Run with pre-defined parameter set
                kr = KernelRidge(alpha=alphaVec[0], kernel='rbf', gamma=sigmaVec[0])
            
            # Fit model
            kr.fit(predictor.reshape(-1,1), predictand.reshape(-1,1))
            
            # Get best parameters
            bestAlpha_kr = kr.best_params_['alpha']
            bestSigma_kr = kr.best_params_['kernel'].length_scale

            # Predict over grid
            kr_fit = kr.predict(predictor_grid.reshape(-1,1))
            
            # Compute derivatives of prediction
            kr_der1 = np.gradient(kr_fit[:,0])
            kr_der2 = np.gradient(kr_der1)
            
            # Estimate decorrelation time KR
            if bestSigma_kr >= 2:
                minDer1 = 0.005 #0.001
            else:
                minDer1 = 0.0
                
            minNormSpread = 0.75
            minNormSpread = 0.75*np.nanmedian(dt.from_dB(predictand)[dt.from_dB(predictor)+dB_shift_hr >= maxLeadTimeHours/2])
            print('Minimum spread to reach:', minNormSpread)
            minNormSpread_dB = dt.to_dB(minNormSpread)
            decorrBool = (kr_der1 <= minDer1) & (kr_der2 < 0) & (kr_fit[:,0] >= minNormSpread_dB)
            decorrIndex_kr = np.where(decorrBool == True)[0]
            
            # Find first local minimum of the derivative
            firstLocalMinimumIndex = argrelextrema(kr_der1, np.less)[0]
            firstLocalMinimumIndex = firstLocalMinimumIndex[0]
            
            if len(decorrIndex_kr) == 0:
                kr_decorr_bad = True
                decorrIndex_kr = len(kr_der1)-1
            else:
                kr_decorr_bad = False
                decorrIndex_kr = decorrIndex_kr[0]
            
            # Take as decorrelation time as the first local minimum before the derivative gets to zero
            criterionLocalMinimum = (decorrIndex_kr > firstLocalMinimumIndex) & (kr_fit[firstLocalMinimumIndex,0] >= minNormSpread_dB) & (bestSigma_kr >= 2)
            if criterionLocalMinimum:
                print('Taking first local minimum as decorrelation time')
                decorrIndex_kr = firstLocalMinimumIndex
            
            # Get decorr time
            if logTime:
                decorr_time_kr = dt.from_dB(predictor_grid[decorrIndex_kr])-dB_shift_hr
            else:
                decorr_time_kr = predictor_grid[decorrIndex_kr]
            
            #### Spherical model fit
            weighting = 1#dt.from_dB(predictor)
            popt, pcov = curve_fit(st.spherical_model, predictor, predictand, sigma=weighting)
            print('Spherical model params:', popt)
            spherical_fit = st.spherical_model(predictor_grid, popt[0], popt[1], popt[2])
            if logTime:
                decorr_time_sph = dt.from_dB(popt[2])-dB_shift_hr
            else:
                decorr_time_sph = popt[2]
            
            #### Exponential model fit
            popt, pcov = curve_fit(st.exponential_model, predictor, predictand, sigma=weighting)
            print('Exponential model params:', popt)
            exponential_fit = st.exponential_model(predictor_grid, popt[0], popt[1], popt[2])
            if logTime:
                decorr_time_exp = dt.from_dB(popt[2])-dB_shift_hr
            else:
                decorr_time_exp = popt[2]
                
            # Estimate decorrelation time simply using a threshold on the KR fit or the raw data
            spreadThreshold = 0.95
            idxDecorr = np.argmax(dt.from_dB(kr_fit) >= spreadThreshold, axis=0)[0]
            if idxDecorr == 0:
                spreadThreshold = 0.8
                idxDecorr = np.argmax(dt.from_dB(kr_fit) >= spreadThreshold, axis=0)[0]
            decorr_time_th = dt.from_dB(predictor_grid[idxDecorr])-dB_shift_hr
            
            if verbosity >= 1:
                print('Lifetime KR          : ', decorr_time_kr, ' h')
                print('Lifetime spherical   : ', decorr_time_sph, ' h')
                print('Lifetime exponential : ', decorr_time_exp, ' h')
                print('Lifetime threshold >=', spreadThreshold, ': ',decorr_time_th, ' h')
            
            #### PLOT THE FITS TO ERROR GROWTH FUNCTIONS
            if plotFits:
                plt.close()
                plt.figure(figsize = (10,10))
                ax1 = plt.subplot(111)
                ax1.scatter(predictor, predictand, marker='o', s=5, color='k')
                
                #ax1.plot(predictor_grid, mars_fit, 'r', label='Multivariate Adaptive Regression Splines (MARS)')
                
                krLabel = r'Kernel Ridge Regression (KR), $\alpha$=' + str(bestAlpha_kr) + r', $\sigma$=' + str(bestSigma_kr)
                p1, = ax1.plot(predictor_grid, kr_fit, 'g', label=krLabel, linewidth=2)
                p2, = ax1.plot(predictor_grid, spherical_fit, 'b', label='Spherical variogram model', linewidth=2)
                p3, = ax1.plot(predictor_grid, exponential_fit, 'r', label='Exponential variogram model', linewidth=2)
                
                # Plot derivatives and decorrelation time
                ax2 = ax1.twinx()
                
                ax2.plot(predictor_grid, kr_der1, 'g--')
                #ax2.plot(predictor_grid, kr_der2*20, 'g:')
                ax2.axvline(x=predictor_grid[decorrIndex_kr], ymin=0.2, color='g')
                ax2.axvline(x=dt.to_dB(decorr_time_sph + dB_shift_hr), ymin=0.2, color='b')
                ax2.axvline(x=dt.to_dB(decorr_time_exp + dB_shift_hr), ymin=0.2, color='r')
                p4 = ax2.axvline(x=dt.to_dB(decorr_time_th + dB_shift_hr), ymin=0.2, color='k')
                
                ax2.axhline(y=0, color='g')
                
                # Labels legend
                p1_label = 'Lifetime KR               : ' + fmt1 % decorr_time_kr + ' h'
                p2_label = 'Lifetime spherical     : ' + fmt1 % decorr_time_sph + ' h'
                p3_label = 'Lifetime exponential: ' + fmt1 % decorr_time_exp + ' h'
                p4_label = 'Lifetime >= ' + fmt2 % spreadThreshold + '      : ' + fmt1 % decorr_time_th + ' h'
                
                plot_lifetimes = [p1,p2,p3,p4]
                labels_lifetimes = [p1_label, p2_label, p3_label, p4_label]
                
                # Plot  legend with lifetimes
                legend_lifetime = plt.legend(plot_lifetimes, labels_lifetimes, loc='upper left', labelspacing=0.1)
                plt.gca().add_artist(legend_lifetime)
                
                # Plot legend of models
                ax1.legend(loc='lower right',labelspacing=0.1)
                ax1.set_xlabel('Lead time, hours', fontsize=20)
                
                # Format X and Y axis
                ax1.set_ylabel(txtYlabel, fontsize=20)
                ax2.set_ylabel('Function derivative', fontsize=20)
                plt.setp(ax1.get_xticklabels(), fontsize=14)
                plt.setp(ax1.get_yticklabels(), fontsize=14) 
                plt.setp(ax2.get_yticklabels(), fontsize=14) 
                plt.xlim([np.min(predictor)-1, np.max(predictor)+1])
                
                if maxLeadTimeHours == 24:
                    xtickLabels = np.array([0.08,0.5,1,2,3,4,5,6,9,12,18,24])
                if maxLeadTimeHours == 48:
                    xtickLabels = np.array([0.08,0.5,1,2,3,4,5,6,9,12,18,24,36,48])
                if maxLeadTimeHours == 96:
                    xtickLabels = np.array([0.08,0.5,1,2,3,4,5,6,9,12,18,24,36,48,72,96])
                xticklocations = dt.to_dB(xtickLabels + dB_shift_hr)
                xtickLabels = dt.dynamic_formatting_floats(xtickLabels)
                ax1.set_xticks(xticklocations)
                xticks = ax1.set_xticklabels(xtickLabels, fontsize=14)
                
                ytickLabels = [0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1,1.2,1.4]
                yticklocations = dt.to_dB(ytickLabels)
                ytickLabels = dt.dynamic_formatting_floats(ytickLabels)
                ax1.set_yticks(yticklocations)
                ax1.set_yticklabels(ytickLabels, fontsize=14)
                
                strTitleLine1 = r'Spread growth for ' + varLabels[variable]
                strTitleLine2 = 'Time series starting in range ' + str(fmt2 % minInitLab) + '-' + str(fmt2 % maxInitLab) + ' (N = ' + str(trajectories.shape[0]) + ')'
                plt.title(strTitleLine1 + '\n' + strTitleLine2, fontsize=22)
                plt.show()
                # fileName = outBaseDir + product + '_' + pltType + '_' + timeStartStr + '-' + timeEndStr +  '0_' + 'Rgt' + str(args.minR) + '_WOLS' + str(args.wols) + '_00005_warGt' + str("%0.1f" % warThreshold) + '_logIMFWAR' + str(int(logIMFWAR)) + '_' + timeAccumMinStr + '.png'
                # print('Saving: ',fileName)
                # plt.savefig(fileName, dpi=300)
                #sys.exit()
            
            ################## PLOTTING ################################################################################
            linewidth=2.0
            labelFontSize = 16
            legendFontSize = 12
            axesTicksFontSize = 14
            
            plt.tick_params(axis='both', which='major', labelsize=axesTicksFontSize)
               
            # Plot growth of spread
            legTxt = ' Range ' + str(fmt2 % minInitLab) + '-' + str(fmt2 % maxInitLab) + ' (N = ' + str(trajectories.shape[0]) + ')'
            
            if pltType == 'spread':
                if (logTime == True) & (logSpread == True):
                    l, = axSP.loglog(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                elif logTime == True:
                    l, = axSP.semilogx(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                elif logSpread == True:
                    l, = axSP.semilogy(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                else:
                    l, = axSP.plot(leadTimesMin/60, spreadArray[:,variable], label=legTxt, linewidth=linewidth)
                # Get lines for second legend
                plot_lines.append(l)
                if kr_decorr_bad == True:
                    strLifetime = 'Lifetime = ' + (fmt1 % decorr_time_exp) + ' h'
                else:
                    strLifetime = 'Lifetime = ' + (fmt1 % decorr_time_kr) + ' h'
                decorr_time_hours.append(strLifetime)
                
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
    
    if pltType == 'spread':
        # Line colors
        colors = [colormap(i) for i in np.linspace(0, 1, len(axSP.lines))]
        for i,j in enumerate(axSP.lines):
            j.set_color(colors[i])
        
        legendFontSize =12
        # Add additional legend with decorrelation time
        legend1 = plt.legend(plot_lines, decorr_time_hours, loc='upper left', fontsize=12, labelspacing=0.1)
        plt.gca().add_artist(legend1)
        
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

    # Plot line of spread saturation
    plt.axhline(1.0, color='k', linestyle='dashed')
    
    # Add labels and title
    plt.xlabel('Lead time [hours]', fontsize=labelFontSize)
    
    if (pltType == 'evolution') & (step == stepEvol):
        plt.ylabel(varLabels[variable], fontsize=labelFontSize)
        strTitle = 'Evolution of ' + varLabels[variable] + ' starting at ' + str(fmt2 % minInitLab) + '-' + str(fmt2 % maxInitLab)
        plt.title(strTitle, fontsize=18)
    if pltType == 'spread':
        plt.ylabel(txtYlabel, fontsize=labelFontSize)
        strTitle = 'Spread growth for ' + varLabels[variable]
        plt.title(strTitle, fontsize=18)
    
    plt.grid(True,which="both", axis='xy')
    # axSP.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1d"))
    # axSP.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1d"))
    axSP.xaxis.set_major_formatter(ticker.FuncFormatter(dt.myLogFormat))
    axSP.yaxis.set_major_formatter(ticker.FuncFormatter(dt.myLogFormat))

toc = time.clock()
print('Total elapsed time: ', toc-tic, ' seconds.')

# Main title
titleStr = 'Growth of spread in the attractor for \n' + product + ': ' + str(timeStamps_datetime[0]) + ' - ' + str(timeStamps_datetime[len(timeStamps_datetime)-1])
plt.suptitle(titleStr, fontsize=20)

# Save figure
fileName = outBaseDir + product + '_' + pltType + '_' + timeStartStr + '-' + timeEndStr +  '0_' + 'Rgt' + str(args.minR) + '_WOLS' + str(args.wols) + '_00005_warGt' + str("%0.1f" % warThreshold) + '_logIMFWAR' + str(int(logIMFWAR)) + '_' + timeAccumMinStr + '.png'
print('Saving: ',fileName)
plt.savefig(fileName, dpi=300)
