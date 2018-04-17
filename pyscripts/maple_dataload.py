#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import getpass
usrName = getpass.getuser()

import argparse
import sys
import os
import csv
import warnings
import math
import numpy as np
import numpy.ma as ma
from scipy import stats
from PIL import Image
import datetime as datetime
from sklearn.externals import joblib

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab
from pylab import get_cmap

# My modules
import gis_base as gis
import time_tools_attractor as ti
import data_tools_attractor as dt
import io_tools_attractor as io

fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"

######################################################################################## 
class Data_Structure(object):
    dirNameArchive = '/scratch/lforesti/maple_data/python-database/'
   
def main(**kwargs):
    # Structures containing the data
    d = Data_Structure()
    
    ################# INPUT ARGS  
    # !!!!!!! This list of arguments should contain all the argumetns used by other scripts importing maple_dataload.py,
    # !!!!!!! also when not used by thsoe scripts !!!!!!
    parser = argparse.ArgumentParser(description='Compute MAPLE archive statistics.')
    parser.add_argument('-start', default='200501010000', type=str, help='Starting date YYYYMMDDHHmm.')
    parser.add_argument('-end', default='200501020000', type=str, help='Ending date YYYYMMDDHHmm.')
    parser.add_argument('-refresh', default=0, type=int, help='Whether to refresh the .npy file with the conditioned archive.')
    parser.add_argument('-refresh_dataset', default=0, type=int, help='Whether to refresh the .npz file with the data matrix for machine learning.')
    parser.add_argument('-model_file', default=None, type=str, help='Which model file to use for predictions (no training needed).')
    parser.add_argument('-clean', default=1, type=int, help='Whether to clean (or using the cleaned) the MAPLE archive by removing bad composite images.')
    parser.add_argument('-hzt', default='none', type=str, help='Which task to do when loading the HZT values at boxes (add, replace, complete)')
    parser.add_argument('-boxsize', default=64, type=int, help='Box size.')
    parser.add_argument('-minIMF', default=0.05, type=float, help='Minimum IMF for conditional statistics.')
    parser.add_argument('-fmt', default='pdf', type=str, help='Figure format.')

    args = parser.parse_args()
    
    boxSize = args.boxsize
    refreshArchive = bool(args.refresh)
    if args.clean == 1:
        d.dataCleaning = 'clean'
    else:
        d.dataCleaning = 'raw'
        
    ############################
    plotHistograms = False

    ########## DOMAIN PARAMETERS 
    dirNameArchive = '/store/msrad/radar/precip_attractor/maple_data/python-database/'
    dirNameArchive = '/scratch/lforesti/maple_data/python-database/'

    ########## ANALYSIS PARAMETERS 
    season_names = ['allyear']#, 'winter', 'spring', 'summer', 'autumn']

    ########## CONDITIONAL STATISTICS PARAMETERS
    thresholdMMHR_lower = args.minIMF
    thresholdMMHR_upper = 100
    conditionalStatistics = True
    
    d.hailStratification = False
    ###############################################################################################################################
    ###############################################################################################################################
    thresholdMMHRStr = str(thresholdMMHR_lower)

    boxSize = args.boxsize
    refreshArchive = bool(args.refresh)
    if args.clean == 1:
        d.dataCleaning = 'clean'
    else:
        d.dataCleaning = 'raw'
        
    ################ GENERATE GEO FEATURES ###############################################
    fileNameGeo = dirNameArchive + 'geo.pkl'
    geo = generate_geo()
    
    ################ COLORMAPS ###########################################################
    print('Preparing colormaps...')
    fileNameCmaps = dirNameArchive + 'cmaps.pkl'
    cmaps = generate_colormaps(fileNameCmaps, d.hailStratification)
    
    ##########################################################################################################
    ######### READ MAPLE DATA ARCHIVE (BOXES) ################################################################
    # Read-in data matrix
    boxSizeStr = '%03i' % boxSize
    newStr = '.new.new'
    fileName = dirNameArchive + 'marray.big.df.' + boxSizeStr + '.Rgt' + thresholdMMHRStr + '_' + d.dataCleaning + newStr + '.npy'

    if (boxSize == 64) & refreshArchive:
        fileNameFull = dirNameArchive + 'marray.big.df.' + boxSizeStr + '_Rgt0.001' + newStr + '.npy'
        fileNameVars = dirNameArchive + 'marray.big.df.' + boxSizeStr + '.Rgt0.001.varNames' + newStr + '.csv'

    else:
        fileNameFull = dirNameArchive + 'marray.big.df.' + boxSizeStr + newStr + '.npy'
        fileNameVars = dirNameArchive + 'marray.big.df.' + boxSizeStr + '.Rgt' + thresholdMMHRStr + '.varNames' + newStr + '.csv'
        
    ti.tic()
    if (os.path.isfile(fileName) == True) & (refreshArchive == False):
        print('Reading:', fileName)
        data = np.load(fileName)
        print('...done.')
    else:
        fileName = fileNameFull
        print('Reading:', fileName)
        if (os.path.isfile(fileName) == True):
            data = np.load(fileName)
        else:
            print('File:', fileName, 'does not exist.')
            sys.exit(1)
        print('...done.')
    ti.toc('to read the data file.')
    
    d.fileName = fileName
    d.fileNameVars = fileNameVars
    
    # Read-in names of variables
    _, variables = io.read_csv(fileNameVars, header=True)
    print(fileNameVars, 'read.')

    # Put indices into dictionary (easier acess using key)
    dictIdx = dict(zip(variables, np.arange(0, len(variables))))
    d.variables = variables
    d.dictIdx = dictIdx
    
    print('----------------------')
    print('Dictionary of variables:')
    print(dictIdx)
    print('----------------------')
    
    ################# READ IN RADAR AVAILABILITY TO CLEAN ARCHIVE ####################################
    ######## Read-in arrays with bad time stamps from radar availability and WAR increments (+ buffer)
    d.timeStampJulian = data[:, dictIdx['time.stamp']].astype(int)        
    if ((refreshArchive == True) | (os.path.isfile(fileName) == False)) & (d.dataCleaning == 'clean'):
        dirNameDataQuality = '/scratch/lforesti/tmp/'

        badTimeStamps = []
        for year in range(2005,2016):
            if year != 2017:
                fileNameQ = dirNameDataQuality + 'AQC-' + str(year) + '_radar-stats-badTimestamps_00005.txt'
            else:
                fileNameQ = dirNameDataQuality + 'AQC-201701010000-201706010000_radar-stats-badTimestamps_00005.txt'
                
            if os.path.isfile(fileNameQ):
                badTimeStampsYear = np.loadtxt(fileNameQ)
                print(fileNameQ, 'read.') 
                print('Bad time stamps', str(year), ':', len(badTimeStampsYear))
                
                badTimeStamps.extend(badTimeStampsYear)
            else:
                print(fileNameQ, 'not found.')     

        badTimeStamps = np.unique(np.array(badTimeStamps)).astype(int).astype('S12') # to avoid having double time stamps 

        # Convert time stamps to MAPLE archive format YYJJJHHMM
        ti.tic()
        badTimeStampsDt = map(ti.timestring2datetime, badTimeStamps)
        badTimeStampsJulian = np.array(map(ti.datetime2juliantimestring, badTimeStampsDt), dtype=int)
        print('----------------------')

        # Remove bad data based on array of time stamps
        idxBad = np.in1d(d.timeStampJulian, badTimeStampsJulian)

        print('N samples before cleaning = ', len(data))
        data = data[~idxBad,:]
        d.timeStampJulian = d.timeStampJulian[~idxBad]
        print('N samples after  cleaning = ', len(data))
        ti.toc('Elapsed time for cleaning data.')
    else:
        print('No data cleaning.')

    print('MAPLE archive period:', data[0, dictIdx['time.stamp']].astype(int), ' to ', data[-1, dictIdx['time.stamp']].astype(int))    

    ########################################################################
    ######## COMPUTE IMF-WAR VARIABLES
    d.IMF_origin = data[:, dictIdx['IMF.rp.1']]
    d.IMF_destination = data[:, dictIdx['IMF.rn.0']]
    d.IMF_destination_realtime = data[:, dictIdx['IMF.rn.1']]

    d.WAR_origin = 100*data[:, dictIdx['WAR.rp.1']]
    d.WAR_destination = 100*data[:, dictIdx['WAR.rn.0']]

    ########################################################################
    # HISTOGRAM of IMF/WAR to select appropriate minimum/maximum rainfall threshold
    if plotHistograms:
        binLims = np.array([0.0001,100])
        binLimsLog = np.log10(binLims)
        binsR = np.linspace(binLimsLog[0], binLimsLog[1], 100)
        binsWAR = np.linspace(0,100,50)

        plt.figure()
        ax = plt.subplot(221)
        ax.hist(d.IMF_destination, 10**binsR)
        ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.title('IMF destination')
        plt.xlabel('Rainrate')

        ax = plt.subplot(222)
        ax.hist(d.WAR_destination, binsWAR)
        plt.title('WAR destination')
        plt.xlabel('WAR')
        
        ax = plt.subplot(223)
        ax.hist(d.IMF_origin, 10**binsR)
        ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.title('IMF origin')
        plt.xlabel('Rainrate')

        ax = plt.subplot(224)
        ax.hist(d.WAR_origin, binsWAR)
        plt.title('WAR origin')
        plt.xlabel('WAR')
        
        plt.tight_layout()
        
        fileName = outBaseDir + 'MAPLE-' + boxSizeStr + '-IMF-WAR-histogram_' + d.dataCleaning + '.' + fig_fmt
        plt.savefig(fileName, dpi=fig_dpi)
        print(fileName, 'saved.')

    ######## CONDITION ON RAINRATE ##################################################
    if conditionalStatistics:
        # Strong conditiona AND
        boolCond = (d.IMF_destination >= thresholdMMHR_lower)*(d.IMF_origin >= thresholdMMHR_lower)
        d.condTxt = 'cond'
    else:
        # Weak conditional OR
        boolCond = (d.IMF_destination >= thresholdMMHR_lower) | (d.IMF_origin >= thresholdMMHR_lower)
        # Set data bewteen 0 and threshold to 0*np
        boolSmallValues = ((d.IMF_destination > 0) & (d.IMF_destination < thresholdMMHR_lower))
        print(np.sum(boolSmallValues == True))
        d.IMF_destination[boolSmallValues == True] = 0.0
        boolSmallValues = ((d.IMF_origin > 0) & (d.IMF_origin < thresholdMMHR_lower))
        d.IMF_origin[boolSmallValues == True] = 0.0
        d.condTxt = 'uncond'
        
    boolCond_upper = (d.IMF_destination > thresholdMMHR_upper) | (d.IMF_origin > thresholdMMHR_upper)
    boolCond[boolCond_upper == True] = False

    print(np.sum(boolCond_upper==True), ' high values removed.')
    print('Percentage of boxes matching rain condition = ', np.sum(boolCond)/len(boolCond)*100, '%')
    print(len(boolCond), ' boxes (total after cleaning).')
    print(np.sum(boolCond==True), ' boxes matching rain condition.')

    ######## Condition dataset (setting invalid to NaN)
    data = data[boolCond,:]
    d.IMF_destination = d.IMF_destination[boolCond]
    d.IMF_origin = d.IMF_origin[boolCond]
    d.IMF_destination_realtime = d.IMF_destination_realtime[boolCond]

    d.WAR_origin = d.WAR_origin[boolCond]
    d.WAR_destination = d.WAR_destination[boolCond]
    d.timeStampJulian = d.timeStampJulian[boolCond]
    
    ####### LOAD HAIL DAYS #####################
    if d.hailStratification:
        print('+++++++++++++++++++++++++++++++++++++++++++')
        print('Growth/decay analysis of severe hail days...')
        fileNameHail = '/users/lforesti/scripts/Major_Hail_Events_2005-2014-Apr-Sep-BZC-80-Switzerland.csv'
        hail_days, head = io.read_csv(fileNameHail,header=True)
        hail_days = np.array(hail_days, dtype=int)
        hail_days[:,0] = hail_days[:,0]*10000
        
        nrDays = 100 # Number of hail days to select (from the most severe)
        severe_hail_days = hail_days[0:nrDays,:]
        severe_hail_days_datetime = ti.timestring_array2datetime_array(severe_hail_days[:,0])
        severe_hail_days_juliantime = map(ti.datetime2juliantimestring, severe_hail_days_datetime)
        severe_hail_days_juliantime = np.array(severe_hail_days_juliantime, dtype=int)

        # SELECT ONLY BOXES FROM SEVERE HAIL DAYS
        timeJulianDay_archive = map(lambda x: int(x/10000), d.timeStampJulian)
        timeJulianDay_hail = map(lambda x: int(x/10000), severe_hail_days_juliantime)
        
        # print(timeJulianDay_archive[0:100], timeJulianDay_hail[0:100])
        idxHail = np.in1d(timeJulianDay_archive, timeJulianDay_hail)
        # idxHail = np.in1d(d.timeStampJulian, severe_hail_days_juliantime)
        
        data[~idxHail, 1:] = np.nan
        d.IMF_destination[~idxHail] = np.nan
        d.IMF_origin[~idxHail] = np.nan
        d.IMF_destination_realtime[~idxHail] = np.nan
        d.WAR_origin[~idxHail] = np.nan
        d.WAR_destination[~idxHail] = np.nan
        # d.timeStampJulian[~idxHail] = np.nan
        print(idxHail.sum(), 'boxes from days with severe hail.')
        print('+++++++++++++++++++++++++++++++++++++++++++')
                
    #####################################################################
    ######### READ-IN FREEZING LEVEL HEIGHT FILES AND ADD TO MAPLE ARCHIVE ################
    d.boolHZT_in_archive = ('HZT_d' in dictIdx.keys()) & ('HZT_o' in dictIdx.keys())

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    if args.hzt != 'none':
        print(np.sum(~np.isnan(data[:,-2:]), axis=0), 'boxes with HZT values found [destination, origin].')
        print('Reading HZT data at boxes and matching...')
        # Reading HZT fiels with boxes and matching with MAPLE archive
        dataDirHZT_base = '/scratch/lforesti/data/'
        task = args.hzt
        data, dictIdx = io.read_hzt_match_maple_archive(data, startTimeStr=args.start, endTimeStr=args.end, dict_colnames=dictIdx, task=task, dataDirHZT_base=dataDirHZT_base, boxSize=boxSize)
        
        # Replace missing with NaN
        missingHZT = (data == -999)
        data[missingHZT] = np.nan
        
        print('----------------------')
        print('New dictionary of variables:')
        print(dictIdx)
        from operator import itemgetter
        variables = sorted(dictIdx.items(), key=itemgetter(1))
        variables = np.array(variables)[:,0]
    else:
        print('HZT data already in MAPLE archive.')

    print(np.sum(~np.isnan(data[:,-2:]), axis=0), 'boxes with HZT values found [destination, origin].')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    #############################################
    ## Save reduced archive for speed up purposes
    if (os.path.isfile(fileName) == False) or (refreshArchive == True) or (args.hzt != 'none'):
        print('Dimensions:', data.shape)
        np.save(fileName, data)
        print(fileName, 'saved.')
        
        # Resave CSV containing additional variable names
        fileNameVars = dirNameArchive + 'marray.big.df.' + boxSizeStr + '.Rgt' + thresholdMMHRStr + '.varNames' + newStr + '.csv'
        io.write_csv(fileNameVars, [variables], header=[])
        print(fileNameVars, 'saved.')

    ############################################
    ########## Define booleans for season stratification
    fileNameSeasons = '/scratch/lforesti/tmp/seasons_bool_' + boxSizeStr + '.Rgt' + thresholdMMHRStr + '_' + d.dataCleaning + '.npz'
    winterBool, springBool, summerBool, autumnBool = generate_bool_seasons(fileNameSeasons, d.timeStampJulian, season_names, refreshArchive)
    d.winterBool = winterBool
    d.springBool = springBool
    d.summerBool = summerBool
    d.autumnBool = autumnBool
        
    ########## GET COORDINATES ##################################################
    # Get grid coordinates (destination box)
    xc_d = data[:, dictIdx['mean.xn.0']]
    yc_d = data[:, dictIdx['mean.yn.0']]

    # Get grid coordinates (origin box)
    xc_o = data[:, dictIdx['mean.xp.1']]
    yc_o = data[:, dictIdx['mean.yp.1']]

    # Transform to Swiss coordinates
    d.x_d, d.y_d = dt.transformStandardToSwiss(xc_d,yc_d)
    d.x_o, d.y_o = dt.transformStandardToSwiss(xc_o,yc_o)

    # Print out coordinate limits
    print('Xc range = ', np.nanmin(xc_d), '-', np.nanmax(xc_d))
    print('Yc range = ', np.nanmin(yc_d), '-', np.nanmax(yc_d),', but top down!')

    print('X range = ', np.nanmin(d.x_d), '-', np.nanmax(d.x_d))
    print('Y range = ', np.nanmin(d.y_d), '-', np.nanmax(d.y_d))
    
    # Compute unique coordinates of boxes
    import pandas as pd
    coordinates = np.vstack((d.x_d, d.y_d)).T
    df = pd.DataFrame(coordinates)
    df = df.drop_duplicates([0,1])
    geo.coordinates_unique = np.array(df.values, dtype=int)

    # CONDITION ON DOMAIN LIMITS
    idxInsideDomainX = (d.x_d >= geo.xlimsKM[0]) & (d.x_d <= geo.xlimsKM[1])
    idxInsideDomainY = (d.y_d >= geo.ylimsKM[0]) & (d.y_d <= geo.ylimsKM[1])
    geo.idxDomain = idxInsideDomainX & idxInsideDomainY
    print(np.sum(geo.idxDomain == True), 'boxes within the reduced domain.')
    
    # Get freezing level height HZT
    if d.boolHZT_in_archive:
        d.HZT_d = data[:, dictIdx['HZT_d']]
        d.HZT_o = data[:, dictIdx['HZT_o']]
    
        isValid = ~np.isnan(d.HZT_d) & ~np.isnan(d.HZT_o) & geo.idxDomain
        percHZT = 100*np.sum(isValid)/np.sum(geo.idxDomain == True)
        print(np.sum(isValid), 'boxes with both HZT_o and HZT_d within the reduced domain.', fmt1 % percHZT, '%')
        
    ########## COMPUTE VARIABLES #####################################################
    print('Computing variables...')
    # Compute displacement vectors
    d.u = d.x_d - d.x_o
    d.v = d.y_d - d.y_o

    ####### Compute growth and decay
    # Linear
    d.GD_lin_IMF = d.IMF_destination - d.IMF_origin
    d.GD_lin_WAR = d.WAR_destination - d.WAR_origin

    d.GD_lin_IMF_dbr = 10.0*np.log10(d.IMF_destination) - 10.0*np.log10(d.IMF_origin)

    # Squared residuals (needed for RMSE decomposition into bias/variance)
    d.GD_squared_IMF_dbr = d.GD_lin_IMF_dbr**2
    d.GD_squared_IMF = (d.IMF_destination - d.IMF_origin)**2

    # Multiplicative
    offsetMMHR = 0.01
    d.GD_log_IMF = 10.0*np.log10((d.IMF_destination + offsetMMHR)/(d.IMF_origin + offsetMMHR))
    offsetPerc = 0.01
    d.GD_log_WAR = 10.0*np.log10((d.WAR_destination + offsetPerc)/(d.WAR_origin + offsetPerc))
    
    boolEul_GD = (d.IMF_destination >= thresholdMMHR_lower) & (d.IMF_destination_realtime >= thresholdMMHR_lower)
    d.IMF_destination_realtime[~boolEul_GD] = np.nan
    d.GD_log_IMF_eul = 10.0*np.log10((d.IMF_destination + offsetMMHR)/(d.IMF_destination_realtime + offsetMMHR))
    d.GD_lin_IMF_eul = d.IMF_destination - d.IMF_destination_realtime
    
    # Get multiplicative GD predicted by MLP
    if 'GD_log_IMF_mlp' in variables:
        d.GD_log_IMF_mlp = data[:, dictIdx['GD_log_IMF_mlp']]
    
    ### Compute polar coordinates
    d.speed = np.sqrt(d.u**2 + d.v**2)
    # radians
    d.theta = np.arctan2(-d.v,-d.u)
    # degrees -180/+180
    d.theta_deg = np.degrees(d.theta)
    # degrees 0/+360
    idxNegative = (np.sign(d.theta_deg) == -1)
    d.theta_deg[idxNegative] = 360 + d.theta_deg[idxNegative]
    # degrees 0/+360, N orientation
    d.theta_deg_N = dt.deg2degN(d.theta_deg)

    if d.boolHZT_in_archive:
        d.pseudofroude_number_d = np.sqrt(d.HZT_d * (d.speed))
        d.pseudofroude_number_o = np.sqrt(d.HZT_o * (d.speed))
    
    ######################################################################################
    ####### Plot HZT values of a set of boxes to check everything is correct
    # jtime_d = 112170900
    # jtime_onext = 112171000
    # bool_jtime_d = (data[:,0] == jtime_d)
    # bool_jtime_onext = (data[:,0] == jtime_onext)

    # print('N boxes =', np.sum(bool_jtime_d))
    # print('N boxes =', np.sum(bool_jtime_onext))

    # Colormap HZT
    # step = 200
    # clevs = np.arange(600,5200+step,step)
    # missingData = 6362.5

    # extend = 'both'
    # cmap, norm = dt.smart_colormap(clevs, name='jet', extend=extend)

    # plt.figure()
    # im = plt.scatter(x_d[bool_jtime_d], y_d[bool_jtime_d], c=data[bool_jtime_d,-2], cmap=cmap, norm=norm)
    # plt.scatter(x_o[bool_jtime_onext], y_o[bool_jtime_onext], c=data[bool_jtime_onext,-1], cmap=cmap, norm=norm, marker='s')

    # plt.title(ti.juliantimestring2datetime(str(jtime_d)))
    # plt.colorbar(im, extend='both')
    # plt.xlim([255, 965])
    # plt.ylim([-160, 480])
    # plt.show()
    
    return(d, geo, cmaps)

#####################################################################################################################################################
#####################################################################################################################################################
def generate_geo(fileNameGeo='/scratch/lforesti/maple_data/python-database/geo.pkl'):
    '''
    Generate geo features (shapefile, DEM, data limits)
    '''
    if (os.path.isfile(fileNameGeo) == True):
        geo = joblib.load(fileNameGeo)
        print(fileNameGeo, 'loaded.')
    else:
        geo = Data_Structure()
        
        # Limits small domain for statistical analysis
        geo.binSpacingKM = 8
        geo.xlimsKM = [400,840]
        geo.ylimsKM = [20,330]

        # Limits large domain for plotting
        geo.xlimsKM_large = [310,910]
        geo.ylimsKM_large = [-100,440]
        
        geo.extent_smalldomain = [geo.xlimsKM[0]*1000, geo.xlimsKM[1]*1000, geo.ylimsKM[0]*1000, geo.ylimsKM[1]*1000]
        geo.extent_largedomain = [geo.xlimsKM_large[0]*1000, geo.xlimsKM_large[1]*1000, geo.ylimsKM_large[0]*1000, geo.ylimsKM_large[1]*1000]
        
        #################### GET DEM ##################################################
        print('Preparing DEM and shapefile...')
        # Read SRTM DEM
        fileNameDEM_SRTM = '/users/lforesti/scripts/dem_proc/dem_merged_projected_clip1000CCS4.tif'
        # fileNameDEM_SRTM = '/users/' + usrName + '/scripts/dem_proc/ritaf_rimini_250m_extendend.tif'
        x_dem,y_dem, geo.demImg,  = gis.gdal_read_raster(fileNameDEM_SRTM)
        x_dem_min = min(x_dem)
        y_dem_min = min(y_dem)
        x_dem_max = max(x_dem)
        y_dem_max = max(y_dem)
        geo.demImg = geo.demImg.astype(float)
        geo.demImg[geo.demImg < -1000] = np.nan

        # Smoothed DEM for countour levels
        from scipy import signal
        kernel_size = 5
        conv_kernel = np.outer(signal.gaussian(kernel_size, kernel_size/4), signal.gaussian(kernel_size, kernel_size/4))
        conv_kernel = conv_kernel/np.sum(conv_kernel)
        geo.demImg_smooth = signal.convolve2d(geo.demImg, conv_kernel, boundary='symm', mode='same')

        # Limits of CCS4 domain (from extent)
        Xmin = 255000
        Xmax = 965000
        Ymin = -160000
        Ymax = 480000
        geo.extent_CCS4 = [Xmin, Xmax, Ymin, Ymax]

        #################### GET SHAPEFILE ##################################################
        # Shapefile name and projections
        dirShp = '/users/' + usrName + '/scripts/shapefiles'
        fileNameShapefile = dirShp + '/CHE_adm/CHE_adm0.shp'
        geo.fileNameShapefile = '/users/lforesti/scripts/shapefiles_proc/CCS4_merged_proj_clip_G05_countries/CCS4_merged_proj_clip_G05_countries.shp' 

        geo.proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84"
        geo.proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
        +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 

        #################### GET RADAR MASK ##################################################
        print('Getting radar mask...')
        # Get one sample radar file for the composite mask
        radar_object = io.read_gif_image('200901010000', product='AQC', minR = 0.1, fftDomainSize = 600, \
            resKm = 1, timeAccumMin = 5, inBaseDir = '/scratch/lforesti/data/', noData = -999.0, cmaptype = 'MeteoSwiss', domain = 'CCS4')
        
        geo.radarMask = radar_object.mask
        geo.radarExtent = radar_object.extent
        
        # Write out geo file
        joblib.dump(geo, fileNameGeo)
        print(fileNameGeo, 'written.')
        
    return(geo)

def generate_colormaps(fileNameCmaps='/scratch/lforesti/maple_data/python-database/cmaps.pkl', hailStratification=False):
    '''
    Generate various colormaps.
    '''
    if (os.path.isfile(fileNameCmaps) == True):
        cmaps = joblib.load(fileNameCmaps)
        print(fileNameCmaps, 'loaded.')
    else:
        cmaps = Data_Structure()
        # Colorscale DEM. Will work only when actual height value are used instead of a grayscale image (CCS4)
        cmaps.clevsDEM = np.array([100, 200, 400, 600, 1000, 1500, 2000, 3000, 4000])
        cmaps.cmapDEM, cmaps.normDEM = dt.smart_colormap(cmaps.clevsDEM, name='gray', extend='both')
        cmaps.alphaDEM = 0.7
        
        # Other colorscales for fields
        cmaps.cmapGD = 'bwr' #'seismic'
        cmaps.alpha=0.65 # Transparency of the field above the DEM (GD, speed, IMF, etc)

        # Linear growth-decay IMF
        step = 0.025
        cmaps.clevsLinIMF = np.arange(-0.25,0.25+step,step)
        cmaps.cmapLinIMF, cmaps.normLinIMF = dt.smart_colormap(cmaps.clevsLinIMF, cmaps.cmapGD, extend='both')

        # Linear growth-decay WAR
        step = 0.1
        cmaps.clevsLinWAR = np.arange(-1,1+step,step)
        cmaps.cmapLinWAR, cmaps.normLinWAR = dt.smart_colormap(cmaps.clevsLinWAR, cmaps.cmapGD, extend='both')

        # Multiplicative growth-decay
        step = 0.2
        cmaps.clevsLog = np.arange(-2.0,2.0+step,step)
        cmaps.cmapLog, cmaps.normLog = dt.smart_colormap(cmaps.clevsLog, cmaps.cmapGD, extend='both')

        # IMF
        if hailStratification == True:
            step = 0.2
            cmaps.clevsIMF = np.arange(0.4,3.0+step,step)
            cmaps.cmapIMF, cmaps.normIMF = dt.colormap_meteoswiss(cmaps.clevsIMF)    
        else:
            step = 0.05
            cmaps.clevsIMF = np.arange(0.3,1.3+step,step)
            cmaps.cmapIMF, cmaps.normIMF = dt.colormap_meteoswiss(cmaps.clevsIMF)

        # Speed
        step = 5
        cmaps.clevsSpeed = np.arange(10,60+step,step)
        # cmaps.cmapSpeed, cmaps.normSpeed = dt.colormap_meteoswiss(cmaps.clevsSpeed)
        # cmaps.cmapSpeed.set_under('w')
        flowSpeed_cmap = 'gnuplot' #'cool' 'summer'
        cmaps.cmapSpeed, cmaps.normSpeed = dt.smart_colormap(cmaps.clevsSpeed, flowSpeed_cmap, extend='both')
        
        # Fraction of bias vs total error
        step = 5
        cmaps.clevsFrac = np.arange(-50,50+step,step)
        cmaps.cmapFrac, cmaps.normFrac = dt.smart_colormap(cmaps.clevsFrac, cmaps.cmapGD, extend='both')
        
        # St.dev of IMF_log_GD
        step = 0.2
        cmaps.clevsStd = np.arange(1.8,4.6+step,step)
        cmaps.cmapStd, cmaps.normStd = dt.smart_colormap(cmaps.clevsStd, cmaps.cmapGD, extend='both')    
        
        # Colormap HZT
        step = 200
        cmaps.clevsHZT = np.arange(600,5200+step,step)
        extend = 'both'
        cmaps.cmapHZT, cmaps.normHZT = dt.smart_colormap(cmaps.clevsHZT, name='jet', extend=extend)
        
        # Base colormap polar histogram
        cmaps.clevsHist = [1,5,10,20,50,100,150,200,250,300,400,500,600,700,800]
        cmaps.cmapHist, cmaps.normHist = dt.smart_colormap(cmaps.clevsHist, 'jet', extend='max')
        
        cmapMask = colors.ListedColormap(['black'])
        cmaps.cmapMask = cmapMask
        
        # Write out cmaps
        joblib.dump(cmaps, fileNameCmaps)
        print(fileNameCmaps, 'written.')
        
    return(cmaps)

def generate_bool_seasons(fileNameSeasons, timeStampJulian, season_names, refreshArchive):
    '''
    generate booleans to stratify statistics by seasons
    '''
    
    conditionSeasons = (('winter' in season_names) | ('spring' in season_names) | ('summer' in season_names) | ('autumn' in season_names))
    
    if conditionSeasons:
        if os.path.isfile(fileNameSeasons) & (refreshArchive == False):
            seasonsBool = np.load(fileNameSeasons)
            winterBool = seasonsBool['winterBool']
            springBool = seasonsBool['springBool']
            summerBool = seasonsBool['summerBool']
            autumnBool = seasonsBool['autumnBool']
            print(fileNameSeasons, 'loaded.')
        else:
            ti.tic()
            timeStampJulianStr = ti.juliantimeInt2juliantimeStr(timeStampJulian)
            yearArrayStr = [u[0:2] for u in timeStampJulianStr]
            yearArray = ti.year2digit_to_year4digit(yearArrayStr)
            julianDayArrayInt = np.array([u[2:5] for u in timeStampJulianStr], dtype=int)
            import calendar
            leapBool = np.array(map(calendar.isleap, yearArray))
            ti.toc('Elapsed time to prepare the time stamps.')

            for season in season_names:
                print('Preparing booleans for ', season)
                if season == 'winter':
                    winterBool = ((julianDayArrayInt >= 335) | (julianDayArrayInt <= 59)) & (leapBool == False)
                if season == 'spring':
                    springBool = ((julianDayArrayInt >= 60) & (julianDayArrayInt <= 151)) & (leapBool == False)
                if season == 'summer':
                    summerBool = ((julianDayArrayInt >= 152) & (julianDayArrayInt <= 243)) & (leapBool == False)
                if season == 'autumn':
                    autumnBool = ((julianDayArrayInt >= 244) & (julianDayArrayInt <= 334)) & (leapBool == False)
                
                ## Leap year
                if season == 'winter':
                    winterBool_l = ((julianDayArrayInt >= 336) | (julianDayArrayInt <= 60)) & (leapBool == True)
                if season == 'spring':
                    springBool_l = ((julianDayArrayInt >= 61) & (julianDayArrayInt <= 152)) & (leapBool == True)
                if season == 'summer':
                    summerBool_l = ((julianDayArrayInt >= 153) & (julianDayArrayInt <= 244)) & (leapBool == True)
                if season == 'autumn':
                    autumnBool_l = ((julianDayArrayInt >= 245) & (julianDayArrayInt <= 335)) & (leapBool == True)
            
            if ('winter' in season_names):
                winterBool = (winterBool | winterBool_l)
            else:
                winterBool = []
            if ('spring' in season_names):
                springBool = (springBool | springBool_l)
            else:
                springBool = []
            if ('summer' in season_names):
                summerBool = (summerBool | summerBool_l)
            else:
                summerBool = []
            if ('autumn' in season_names):
                autumnBool = (autumnBool | autumnBool_l)
            else:
                autumnBool = []
                
            np.savez(fileNameSeasons, winterBool=winterBool[boolCond], springBool=springBool[boolCond], summerBool=summerBool[boolCond], autumnBool=autumnBool[boolCond])
            print(fileNameSeasons, 'saved.')
            
        #### Print out nr boxes per season
        print('-------------------------------------------')
        
        if ('winter' in season_names):
            print('Nr boxes winter:', np.sum(winterBool[boolCond]))
        if ('spring' in season_names):
            print('Nr boxes spring:', np.sum(springBool[boolCond]))
        if ('summer' in season_names):
            print('Nr boxes summer:', np.sum(summerBool[boolCond]))
        if ('autumn' in season_names):
            print('Nr boxes autumn:', np.sum(autumnBool[boolCond]))

        print('-------------------------------------------')
    
        return(winterBool, springBool, summerBool, autumnBool)
    else:
        return([],[],[],[])
    
def get_colormap_and_title(cmaps, whichPlot, statistics=None):
    if (whichPlot == 'GD_lin_IMF') or (whichPlot == 'GD_lin_IMF_eul'):
        clevs = cmaps.clevsLinIMF
        norm = cmaps.normLinIMF
        cmap = cmaps.cmapLinIMF
        titleStr = 'MAP growth-decay'
        ext = 'both'
        cbTitle = r'   mm hr$^{-1}$'
    elif (whichPlot == 'GD_log_IMF') or (whichPlot == 'GD_log_IMF-WAR') or (whichPlot == 'GD_log_IMF_eul') or (whichPlot == 'GD_log_IMF_mlp') or (whichPlot == 'GD_log_IMF_mlp_diff'):
        clevs = cmaps.clevsLog
        norm = cmaps.normLog
        cmap = cmaps.cmapLog
        titleStr = 'MAP growth-decay'
        ext = 'both'
        cbTitle = '   dB'
        if (statistics == 'std'):
            clevs = cmaps.clevsStd
            norm = cmaps.normStd
            cmap = cmaps.cmapStd
            titleStr = r'MAP growth-decay (st. dev.)'
        if (statistics == 'frac'):
            clevs = cmaps.clevsFrac
            norm = cmaps.normFrac
            cmap = cmaps.cmapFrac
            titleStr = r'% bias$^2$ / total var.'
            ext = 'both'
            cbTitle = '   %'
    elif whichPlot == 'GD_lin_WAR':
        clevs = cmaps.clevsLinWAR
        norm = cmaps.normLinWAR
        cmap = cmaps.cmapLinWAR
        titleStr = 'WAR growth-decay' 
        ext = 'both'
        cbTitle = r'   mm hr$^{-1}$'
    elif whichPlot == 'GD_log_WAR':
        clevs = cmaps.clevsLog
        norm = cmaps.normLog
        cmap = cmaps.cmapLog
        titleStr = r'WAR growth-decay' 
        ext = 'both'
        cbTitle = '   dB'
    elif whichPlot == 'speed':
        clevs = cmaps.clevsSpeed
        norm = cmaps.normSpeed
        cmap = cmaps.cmapSpeed
        titleStr = r'Box speed'
        ext = 'both'
        cbTitle = r'   km h$^{-1}$'
    elif whichPlot == 'histogram':
        clevs = cmaps.clevsHist
        norm = cmaps.normHist
        cmap = cmaps.cmapHist
        titleStr = r'Flow rose'
        ext = 'max'
        cbTitle = '   count'
    elif (whichPlot == 'imf_o') or (whichPlot == 'imf_d'):
        clevs = cmaps.clevsIMF
        norm = cmaps.normIMF
        cmap = cmaps.cmapIMF
        ext = 'max'
        cbTitle = r'   mm h$^{-1}$'
        if (whichPlot == 'imf_o'):
            titleStr = r'MAP$_o$'
        if (whichPlot == 'imf_d'):
            titleStr = r'MAP$_d$'
    elif (whichPlot == 'hzt_o') or (whichPlot == 'hzt_d'):
        clevs = cmaps.clevsHZT
        norm = cmaps.normHZT
        cmap = cmaps.cmapHZT
        if (whichPlot == 'hzt_o'):
            titleStr = r'HZT$_o$'
        if (whichPlot == 'hzt_d'):
            titleStr = r'HZT$_d$'
        ext = 'both'
        cbTitle = r'   m.a.s.l.'
    else:
        print(whichPlot, 'not available in get_colormap_and_title.')
        print('Set the clevs, norm and cmap parameters after calling the function.')
        clevs=[]
        norm=[] 
        cmap=[]
        titleStr = r'uncond. MAP$_d$'
        ext='max'
        cbTitle=r'   mm h$^{-1}$'
        
    return(clevs, norm, cmap, titleStr, ext, cbTitle)
    
def get_stat_function(statistics, perc=None): 
    # Define personalized functions for binned_statistics
    if (statistics == 'mean') | (statistics == 'median'):
        stat_func = statistics
    elif statistics == 'std':
        stat_func = np.std
    elif statistics == 'mse':
        stat_func = lambda x: np.mean(x**2)
    elif statistics == 'frac':
        # stat_func = lambda x: 100.0*np.abs(np.mean(x))/(np.abs(np.mean(x)) + np.std(x)) Wrong decomposition
        stat_func = lambda x: np.sign(np.mean(x))*100.0*np.mean(x)**2/(np.mean(x)**2 + np.std(x)**2)
    elif statistics == 'cv':
        stat_func_ratio = lambda x: np.std(x)/np.mean(x)
        stat_func_diff = lambda x: np.std(x) - np.abs(np.mean(x)) # To compute the CV for an already multiplicative variable (GD)
    elif statistics == 'iqr':
        stat_func = lambda x: np.percentile(x,75) - np.percentile(x,25)
    elif statistics == 'percentile':
        if perc == None:
            print('Do not forget to pass the wanted percentile. I will use 50 by default...')
            perc = 50
        stat_func = lambda x: np.percentile(x, perc)
    else:
        print('Wrong statistics asked:', statistics)
        sys.exit(1)
    return(stat_func)