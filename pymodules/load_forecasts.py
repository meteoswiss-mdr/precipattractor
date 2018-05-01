from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import datetime
from netCDF4 import Dataset

import netCDF4
import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt
import nowcasting as nw

import run_fieldextra_c1 as rf1
import run_fieldextra as rfe


def produce_radar_observation_with_accumulation(startValidTimeStr, endValidTimeStr, newAccumulationMin=10, domainSize=512, product='RZC',rainThreshold=0.08):

    # base accumulation is 5 min
    baseAccumMin = 5
    accumFactor = int(newAccumulationMin/baseAccumMin)
    
    # datetime format
    startValidTime = ti.timestring2datetime(startValidTimeStr)
    endValidTime = ti.timestring2datetime(endValidTimeStr)
    leadTimeMin = int((endValidTime - startValidTime).total_seconds()/60)

    # start to compute the correct accumulation at t0
    if newAccumulationMin>baseAccumMin:
        startTimeToPass = startValidTime - datetime.timedelta(minutes=newAccumulationMin-baseAccumMin)
    else: 
        startTimeToPass = startValidTime
    startTimeToPassStr = ti.datetime2timestring(startTimeToPass)
    
    with np.errstate(invalid='ignore'):
        radar_observations_5min, radar_mask_5min, timestamps, r = nw.get_radar_observations(startTimeToPassStr, leadTimeMin+newAccumulationMin-baseAccumMin, product=product, rainThreshold = 0)

    # convert to mm for computing accumulations
    radar_observations_5min = radar_observations_5min/60*baseAccumMin
    
    # aggregate to new accumulation
    if newAccumulationMin>baseAccumMin:
        radar_observations_new = nw.aggregate_in_time(radar_observations_5min,timeAccumMin=newAccumulationMin,type='sum')
        radar_mask_new = nw.aggregate_in_time(radar_mask_5min,timeAccumMin=newAccumulationMin,type='nansum')
        radar_mask_new[radar_mask_new==0] = np.nan
        radar_mask_new[radar_mask_new>0] = 1
        timestamps = timestamps[accumFactor-1::accumFactor]
    else:
        radar_observations_new = radar_observations_5min
        radar_mask_new = radar_mask_5min  
        
    # convert to mm/h
    radar_observations_new = radar_observations_new/newAccumulationMin*60
    
    # Apply rain threshold
    radar_observations_new[radar_observations_new<=rainThreshold] = 0
   
    # [time,y,x]
    radar_observations_new = np.rollaxis(radar_observations_new,2,0)
    radar_mask_new = np.rollaxis(radar_mask_new,2,0)
    
    return radar_observations_new,radar_mask_new,timestamps
    
def get_radar_extrapolation(startValidTimeStr,endValidTimeStr, newAccumulationMin=10, domainSize=512, rainThreshold=0.08, product='RZC', outBaseDir='/scratch/ned/data/'):

    # datetime format
    startValidTime = ti.timestring2datetime(startValidTimeStr)
    endValidTime = ti.timestring2datetime(endValidTimeStr)
    leadTimeMin = int((endValidTime - startValidTime).total_seconds()/60)
    
    # Check if the nc file already exists
    year = startValidTime.year
    yearStr =  str(year)[2:4]
    julianDay = startValidTime.timetuple().tm_yday
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    outDir = outBaseDir + startValidTime.strftime("%Y") + '/' + startValidTime.strftime("%y") + julianDayStr + '/'
    fcstName = 'radar-extrapolation_' + startValidTime.strftime("%Y%m%d%H%M") + '_' + str(int(leadTimeMin/60)) + 'hours'
    fcstFile = r'%s' % (outDir + fcstName + '.nc')
    
    # base accumulation is 5 min
    baseAccumMin = 5
    accumFactor = int(newAccumulationMin/baseAccumMin)

    # if not produce forecasts
    if os.path.isfile(fcstFile) == False:
        
        # produce 5-min radar extrapolation
        radar_extrapolation_5min, timestamps_5min = nw.radar_extrapolation(startValidTimeStr,leadTimeMin,finalDomainSize=domainSize,product=product,rainThreshold=rainThreshold)
        if product=='RZC':
            r = io.read_bin_image(startValidTimeStr, fftDomainSize=domainSize, inBaseDir = '/scratch/ned/data/')
        else:
            r = io.read_gif_image(startValidTimeStr, fftDomainSize=domainSize, inBaseDir = '/scratch/ned/data/')
        
        if accumFactor>1:
            # convert to mm 
            radar_extrapolation_5min = radar_extrapolation_5min/60*baseAccumMin
            
            # aggregate to new accumulation (to match COSMO1 resolution)
            radar_extrapolation_new = nw.aggregate_in_time(radar_extrapolation_5min,timeAccumMin=newAccumulationMin,type='sum')
            timestamps_new = timestamps_5min[accumFactor-1::accumFactor]
            
            # convert to mm/h
            radar_extrapolation_new = radar_extrapolation_new/newAccumulationMin*60
             
            # get observations at t0 [mm/h]
            radar_observations_t0, _, _ = produce_radar_observation_with_accumulation(startValidTimeStr, startValidTimeStr, newAccumulationMin, domainSize=domainSize, product=product)
            radar_observations_t0 = np.squeeze(radar_observations_t0)[:,:,None]
        else:
            # no need to aggregate forecasts
            radar_extrapolation_new = radar_extrapolation_5min
            timestamps_new = timestamps_5min
        
            # get observations at t0
            radar_observations_t0 = r.rainrate[:,:,None]
            
        # add observation for t0
        radar_extrapolation_new = np.concatenate((radar_observations_t0,radar_extrapolation_new),axis=2)
        timestamps_new.insert(0,startValidTime)
        timestamps_new = np.array(timestamps_new)
            
        # [time,y,x]
        radar_extrapolation_new = np.rollaxis(radar_extrapolation_new,2,0)

        # save netcdf 
        Xcoords = r.subXcoords
        Ycoords = r.subYcoords
        save_3darray_netcdf(fcstFile, radar_extrapolation_new, 'radar_extrapolation',\
                        timestamps_new,Xcoords,Ycoords)

    print('Read: ' + fcstFile)
    # Now read the NetCDF file
    radar_extrapolation, timestamps, Xcoords, Ycoords = load_3darray_netcdf(fcstFile)
    
    testplot=False
    if testplot:
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(radar_extrapolation[t,:,:],interpolation ='nearest',vmin=0,vmax=65, extent=[Xcoords.min(), Xcoords.max(), Ycoords.min(), Ycoords.max()])
                plt.title(timestamps[t])
                plt.pause(1)

    return radar_extrapolation, timestamps, Xcoords, Ycoords
 
def get_ensemble_radar_extrapolation(startValidTimeStr,endValidTimeStr, NumberMembers = 2, NumberLevels = 1, newAccumulationMin=10, rainThreshold = 0.08, domainSize=512, local_level = 0, seed = 42, product='RZC', outBaseDir='/scratch/ned/data/'):

    # datetime format
    startValidTime = ti.timestring2datetime(startValidTimeStr)
    endValidTime = ti.timestring2datetime(endValidTimeStr)
    leadTimeMin = int((endValidTime - startValidTime).total_seconds()/60)
    
    # Check if the nc file already exists
    year = startValidTime.year
    yearStr =  str(year)[2:4]
    julianDay = startValidTime.timetuple().tm_yday
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    outDir = outBaseDir + startValidTime.strftime("%Y") + '/' + startValidTime.strftime("%y") + julianDayStr + '/'
    fcstName = 'ensemble-radar-extrapolation_' + startValidTime.strftime("%Y%m%d%H%M") + '_' + str(int(leadTimeMin/60)) + 'hours' + '_' + str(NumberMembers) + 'members' + '_' + str(NumberLevels) + 'levels' + '_' + str(newAccumulationMin) + 'min' + '_' + product + '_' + str(domainSize) + 'km' + '_' + str(local_level) + 'local_seed' + str(seed)
    fcstNameMask =  fcstName + '_mask'
    fcstFile = r'%s' % (outDir + fcstName + '.nc')
    fcstFileMask = r'%s' % (outDir + fcstNameMask + '.nc')
    
    # base accumulation is 5 min
    baseAccumMin = 5
    accumFactor = int(newAccumulationMin/baseAccumMin)

    # if not produce forecasts
    if os.path.isfile(fcstFile) == False:
        
        # produce 5-min radar extrapolation
        radar_extrapolation_5min, timestamps_5min, radarMask_5min = nw.probabilistic_radar_extrapolation(startValidTimeStr,leadTimeMin,finalDomainSize=domainSize,NumberMembers=NumberMembers,NumberLevels=NumberLevels,product=product,local_level=local_level,seed=seed)
        if product=='RZC':
            r = io.read_bin_image(startValidTimeStr, fftDomainSize=domainSize, product=product, inBaseDir = '/scratch/ned/data/')
        else:
            r = io.read_gif_image(startValidTimeStr, fftDomainSize=domainSize, product=product, inBaseDir = '/scratch/ned/data/')
        # print(radarMask_5min.shape)
        if accumFactor>1:
            # convert to mm 
            radar_extrapolation_5min = radar_extrapolation_5min/60*baseAccumMin
            
            # aggregate to new accumulation (to match COSMO1 resolution)
            for m in xrange(NumberMembers):
                radar_extrapolation_new_member = nw.aggregate_in_time(radar_extrapolation_5min[:,:,:,m],timeAccumMin=newAccumulationMin,type='sum')
                if m==0:
                    radar_extrapolation_new = np.zeros((radar_extrapolation_new_member.shape[0],radar_extrapolation_new_member.shape[1],radar_extrapolation_new_member.shape[2],NumberMembers))
                radar_extrapolation_new[:,:,:,m] = radar_extrapolation_new_member
            
            timestamps_new = timestamps_5min[accumFactor-1::accumFactor]
            radarMask_new = radarMask_5min[:,:,accumFactor-1::accumFactor]
            # print(radarMask_new.shape)
            # convert to mm/h
            radar_extrapolation_new = radar_extrapolation_new/newAccumulationMin*60
             
            # get observations at t0 [mm/h]
            radar_observations_t0, _, _ = produce_radar_observation_with_accumulation(startValidTimeStr, startValidTimeStr, newAccumulationMin, domainSize, product=product)
            radar_observations_t0 = np.squeeze(radar_observations_t0)
            radarmask_t0 = r.mask.copy()
            radarmask_t0 = np.array(np.isnan(radarmask_t0),dtype=int)
            # print(radarmask_t0.shape)
        else:
            # no need to aggregate forecasts
            radar_extrapolation_new = radar_extrapolation_5min
            timestamps_new = timestamps_5min
            radarMask_new = radarMask_5min
        
            # get observations at t0
            radar_observations_t0 = r.rainrate.copy()
            radarmask_t0 = r.mask.copy()
            radarmask_t0 = np.array(np.isnan(radarmask_t0),dtype=int)
            
        radar_observations_t0_allmembers = np.zeros((radar_observations_t0.shape[0],radar_observations_t0.shape[1],NumberMembers))
        for m in xrange(NumberMembers):
            radar_observations_t0_allmembers[:,:,m] = radar_observations_t0.copy()
            
        radar_observations_t0_allmembers = radar_observations_t0_allmembers[:,:,None,:]
        radarmask_t0 = radarmask_t0[:,:,None]
        # print(radarmask_t0.shape)
        # add observation for t0
        radar_extrapolation_new = np.concatenate((radar_observations_t0_allmembers,radar_extrapolation_new),axis=2)
        radarMask_new = np.concatenate((radarmask_t0,radarMask_new),axis=2)
        timestamps_new.insert(0,startValidTime)
        timestamps_new = np.array(timestamps_new)
        
        # [time,member,y,x]
        radar_extrapolation_new = np.rollaxis(radar_extrapolation_new,2,0)
        radar_extrapolation_new = np.rollaxis(radar_extrapolation_new,3,1)
        radarMask_new = np.rollaxis(radarMask_new,2,0)
        
        # Apply rain threshold
        radar_extrapolation_new[radar_extrapolation_new<=rainThreshold] = 0
        
        # save netcdf 
        Xcoords = r.subXcoords
        Ycoords = r.subYcoords
        save_4darray_netcdf(fcstFile, radar_extrapolation_new, 'ensemble_radar_extrapolation',\
                        timestamps_new,NumberMembers,Xcoords,Ycoords)
        save_3darray_netcdf(fcstFileMask, radarMask_new, 'ensemble_radar_extrapolation_mask',\
                        timestamps_new,Xcoords,Ycoords)

    print('Read: ' + fcstFile)
    # Now read the NetCDF file
    ensemble_radar_extrapolation, timestamps, members, Ycoords, Xcoords = load_4darray_netcdf(fcstFile)
    if os.path.isfile(fcstFileMask):
        ensemble_radar_extrapolation_mask, _, _, _ = load_3darray_netcdf(fcstFileMask)
    else: 
        print('Radar mask file not found.')
        ensemble_radar_extrapolation_mask = np.zeros((timestamps.size,ensemble_radar_extrapolation.shape[2],ensemble_radar_extrapolation.shape[3]))*np.nan
        
    testplot=False
    if testplot:
        r = io.read_gif_image(startValidTimeStr)
        nmember=0
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(ensemble_radar_extrapolation[t,nmember,:,:],interpolation ='nearest',norm=r.norm,cmap=r.cmap)
                plt.title(timestamps[t])
                plt.pause(1)

    return ensemble_radar_extrapolation, timestamps, members, Xcoords, Ycoords, ensemble_radar_extrapolation_mask
    
def get_cosmo1(startValidTimeStr, endValidTimeStr,domainSize=512, leadTimeMin = 12*60, outBaseDir='/scratch/ned/data/',rainThreshold=0.08):
        
    # Get most recent run
    analysisTimeStr = rf1.find_nearest_run_cosmo1(startValidTimeStr)
    analysisTime = ti.timestring2datetime(analysisTimeStr)
        
    # Check if the nc file already exists
    year = analysisTime.year
    yearStr =  str(year)[2:4]
    julianDay = analysisTime.timetuple().tm_yday
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    outDir = outBaseDir + analysisTime.strftime("%Y") + '/' + analysisTime.strftime("%y") + julianDayStr + '/'
    fcstName = analysisTime.strftime("%Y%m%d%H%M") + '_' + str(int(leadTimeMin/60)) + 'hours'
    fcstFile = r'%s' % (outDir + 'cosmo-1_TOT_PREC_' + fcstName + '.nc')
    # if not call fieldextra
    if os.path.isfile(fcstFile) == False:
        rf1.run_fieldextra_c1(analysisTimeStr,leadTimeMin,outBaseDir=outBaseDir)
    print('Read: ' + fcstFile)
    
    # Now read the NetCDF file
    cosmo1_data, timestamps, Xcoords, Ycoords = load_3darray_netcdf(fcstFile)
    
    # exclude first time step (it's all NaNs!)
    cosmo1_data=cosmo1_data[1:,:,:]
    timestamps=timestamps[1:]
    
    # convert to mm/h
    cosmo1_data = cosmo1_data*6
    # flip and extract middle domain
    new_data = np.zeros((cosmo1_data.shape[0],domainSize,domainSize))
    for i in range(cosmo1_data.shape[0]):
        # flip frames
        cosmo1_data[i,:,:] = np.flipud(cosmo1_data[i,:,:])
        # cut domain
        if domainSize>0:
            new_data[i,:,:] = dt.extract_middle_domain(cosmo1_data[i,:,:], domainSize, domainSize)
        else:
            new_data[i,:,:] = cosmo1_data[i,:,:]
    cosmo1_data = new_data
    
    # Apply rain threshold
    cosmo1_data[cosmo1_data<=rainThreshold] = 0 
    
    # Get coordinates of reduced domain
    if domainSize>0:
        extent = dt.get_reduced_extent(Xcoords.shape[0], Ycoords.shape[0], domainSize, domainSize)
        Xmin = Xcoords[extent[0]]
        Ymin = Ycoords[extent[1]]
        Xmax = Xcoords[extent[2]]
        Ymax = Ycoords[extent[3]]    
    extent = (Xmin, Xmax, Ymin, Ymax)
    subXcoords = np.arange(Xmin,Xmax,1000)
    subYcoords = np.arange(Ymin,Ymax,1000) 

    # Extract timestamps
    idxKeep1 = timestamps >= ti.timestring2datetime(startValidTimeStr)
    idxKeep2 = timestamps <= ti.timestring2datetime(endValidTimeStr)
    cosmo1_data = cosmo1_data[idxKeep1*idxKeep2,:,:]
    timestamps = timestamps[idxKeep1*idxKeep2]
    
    testplot=False
    if testplot:
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(cosmo1_data[t,:,:],interpolation ='nearest',vmin=0,vmax=65, extent=[subXcoords.min(), subXcoords.max(), subYcoords.min(), subYcoords.max()])
                plt.title(timestamps[t])
                plt.pause(1)
    
    return cosmo1_data, timestamps, subXcoords, subYcoords   
   
def get_cosmoE(startValidTimeStr, endValidTimeStr, analysisTimeStr = [], domainSize=512, leadTimeMin = 24*60, outBaseDir='/scratch/ned/data/',rainThreshold=0.08, latencyTimeMin=100, overwrite = False):

    # Get most recent run
    if len(analysisTimeStr)==0: 
        analysisTimeStr = rfe.find_nearest_forecast(startValidTimeStr, 'cosmo-e', lat_timeMin = latencyTimeMin)
    analysisTime = ti.timestring2datetime(analysisTimeStr)
    leadtimeHrs = int (( ti.timestring2datetime(endValidTimeStr) - analysisTime ).total_seconds()/3600 )
    
    # # Check if the nc file already exists
    # year = analysisTime.year
    # yearStr =  str(year)[2:4]
    # julianDay = analysisTime.timetuple().tm_yday
    # julianDayStr = '%03i' % julianDay
    # yearJulianStr = yearStr + julianDayStr
    # outDir = outBaseDir + analysisTime.strftime("%Y") + '/' + analysisTime.strftime("%y") + julianDayStr + '/'
    # # fcstName = analysisTime.strftime("%Y%m%d%H%M") + '_' + str(int(leadTimeMin/60)) + 'hours'
    # fcstName = analysisTimeStr + '_' + startValidTimeStr + '_' + endValidTimeStr
    # fcstFile = r'%s' % (outDir + 'cosmo-E_TOT_PREC_' + fcstName + '.nc')
    # # if not call fieldextra
    # if os.path.isfile(fcstFile) == False:
        # rfe.run_fieldextra_ce(analysisTimeStr,startValidTimeStr,endValidTimeStr,outBaseDir=outBaseDir)
        
    outFile = rfe.run_fieldextra_forecast(analysisTimeStr, leadtimeHrs, fieldName='TOT_PREC', outBaseDir = '/scratch/ned/data/', modelName='cosmo-e', deltaMin = 60, overwrite = overwrite)
        
    print('Read: ' + outFile)
    
    # Now read the NetCDF file
    cosmoe_data, timestamps, members, Ycoords, Xcoords = load_4darray_netcdf(outFile)

    # exclude first time step (it's all NaNs!)
    # cosmoe_data=cosmoe_data[1:,:,:,:]
    # timestamps=timestamps[1:]
    
    # flip and extract middle domain
    new_data = np.zeros((cosmoe_data.shape[0], cosmoe_data.shape[1],domainSize,domainSize))
    for i in xrange(cosmoe_data.shape[0]):
        for m in xrange(cosmoe_data.shape[1]):
            # flip frames
            cosmoe_data[i,m,:,:] = np.flipud(cosmoe_data[i,m,:,:])
            # cut domain
            if domainSize>0:
                new_data[i,m,:,:] = dt.extract_middle_domain(cosmoe_data[i,m,:,:], domainSize, domainSize)
            else:
                new_data[i,m,:,:] = cosmoe_data[i,m,:,:]
    cosmoe_data = new_data
    
    # Apply rain threshold
    cosmoe_data[cosmoe_data<=rainThreshold] = 0 
    
    # Get coordinates of reduced domain
    if domainSize>0:
        extent = dt.get_reduced_extent(Xcoords.shape[0], Ycoords.shape[0], domainSize, domainSize)
        Xmin = Xcoords[extent[0]]
        Ymin = Ycoords[extent[1]]
        Xmax = Xcoords[extent[2]]
        Ymax = Ycoords[extent[3]]    
    extent = (Xmin, Xmax, Ymin, Ymax)
    subXcoords = np.arange(Xmin,Xmax,1000)
    subYcoords = np.arange(Ymin,Ymax,1000) 

    # Extract timestamps
    idxKeep1 = timestamps >= ti.timestring2datetime(startValidTimeStr)
    idxKeep2 = timestamps <= ti.timestring2datetime(endValidTimeStr)
    cosmoe_data = cosmoe_data[idxKeep1*idxKeep2,:,:,:]
    timestamps = timestamps[idxKeep1*idxKeep2]
    
    testplot=False
    if testplot:
        nmember=0
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(cosmoe_data[t,nmember,:,:],interpolation ='nearest',vmin=0,vmax=65, extent=[subXcoords.min(), subXcoords.max(), subYcoords.min(), subYcoords.max()])
                plt.title(timestamps[t])
                plt.pause(1)
    
    return cosmoe_data, timestamps, subXcoords, subYcoords   

def get_lagged_cosmo1(startValidTimeStr, endValidTimeStr, domainSize=512, leadTimeMin = 6*60, outBaseDir='/scratch/ned/data/', rainThreshold=0.08):
  
    # get all individual forecasts' starting times
    latencyTimeHr = 1#100/60
    maxLeadTimeHr=33
    fcstStarts,fcstMembers,fcstLeadTimeStartsMin,fcstLeadTimeStopsMin = rf1.get_lagged_ensemble_members(startValidTimeStr, endValidTimeStr, latencyTimeHr, maxLeadTimeHr)
    nmembers = len(fcstMembers)
    
    # if necessary, extract them with fieldextra
    filesOut = []
    analysisTimes = []
    for n in xrange(nmembers):
        print('----------------------------------------------------')
        print(fcstMembers[n] + ', analysis time: ' + fcstStarts[n].strftime("%Y%m%d%H%M%S") + ', ' + str(fcstLeadTimeStartsMin[n]/60) + ' to ' + str(fcstLeadTimeStopsMin[n]/60) + ' hours.')
        
        # Check if the nc file already exists
        year = fcstStarts[n].year
        yearStr =  str(year)[2:4]
        julianDay = fcstStarts[n].timetuple().tm_yday
        julianDayStr = '%03i' % julianDay
        yearJulianStr = yearStr + julianDayStr
        outDir = outBaseDir + fcstStarts[n].strftime("%Y") + '/' + fcstStarts[n].strftime("%y") + julianDayStr + '/'
        
        it_exists = False
        LeadTimeToCheck = fcstLeadTimeStopsMin[n]/60
        while (it_exists == False) and (LeadTimeToCheck <= maxLeadTimeHr):
            fcstName = fcstStarts[n].strftime("%Y%m%d%H%M") + '_' + str(int(LeadTimeToCheck)) + 'hours'
            fcstFile = r'%s' % (outDir + 'cosmo-1_TOT_PREC_' + fcstName + '.nc')
            if os.path.isfile(fcstFile) == False:
                LeadTimeToCheck += 1
            else:
                it_exists = True
                break
        
        if it_exists == False:
            LeadTimeToGetHrs = np.minimum(maxLeadTimeHr, np.ceil(( fcstLeadTimeStopsMin[n]/60 )/12)*12)
            fcstFile = rf1.run_fieldextra_c1(fcstStarts[n].strftime("%Y%m%d%H%M"),LeadTimeToGetHrs*60,outBaseDir=outBaseDir)
        
        print('read: ' + fcstFile)
        
        # Now read the NetCDF file
        cosmo1_data, timestamps, Xcoords, Ycoords = load_3darray_netcdf(fcstFile)
        
        # exclude first time step (it's all NaNs!)
        cosmo1_data=cosmo1_data[1:,:,:]
        timestamps=timestamps[1:]
        
        # convert to mm/h
        cosmo1_data = cosmo1_data*6
        # flip and extract middle domain
        new_data = np.zeros((cosmo1_data.shape[0],domainSize,domainSize))
        for i in range(cosmo1_data.shape[0]):
            # flip frames
            cosmo1_data[i,:,:] = np.flipud(cosmo1_data[i,:,:])
            # cut domain
            if domainSize>0:
                new_data[i,:,:] = dt.extract_middle_domain(cosmo1_data[i,:,:], domainSize, domainSize)
            else:
                new_data[i,:,:] = cosmo1_data[i,:,:]
        cosmo1_data = new_data
        
        # Apply rain threshold
        cosmo1_data[cosmo1_data<=rainThreshold] = 0 
        
        
        # Extract timestamps
        idxKeep1 = timestamps >= ti.timestring2datetime(startValidTimeStr)
        idxKeep2 = timestamps <= ti.timestring2datetime(endValidTimeStr)
        cosmo1_data = cosmo1_data[idxKeep1*idxKeep2,:,:]
        timestamps = timestamps[idxKeep1*idxKeep2]
        
        # Build 4D array with all members
        if n==0:
            cosmo1_lagged_data = np.zeros((cosmo1_data.shape[0],nmembers,cosmo1_data.shape[1],cosmo1_data.shape[2]))
        cosmo1_lagged_data[:,n,:,:] = cosmo1_data.copy()


    # Get coordinates of reduced domain
    if domainSize>0:
        extent = dt.get_reduced_extent(Xcoords.shape[0], Ycoords.shape[0], domainSize, domainSize)
        Xmin = Xcoords[extent[0]]
        Ymin = Ycoords[extent[1]]
        Xmax = Xcoords[extent[2]]
        Ymax = Ycoords[extent[3]]    
    extent = (Xmin, Xmax, Ymin, Ymax)
    subXcoords = np.arange(Xmin,Xmax,1000)
    subYcoords = np.arange(Ymin,Ymax,1000) 
    
    testplot=False
    if testplot:
        nmember=0
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(cosmo1_lagged_data[t,nmember,:,:],interpolation ='nearest',vmin=0,vmax=65, extent=[subXcoords.min(), subXcoords.max(), subYcoords.min(), subYcoords.max()])
                plt.title(timestamps[t])
                plt.pause(1)
    
    return cosmo1_lagged_data, timestamps, subXcoords, subYcoords       

def get_cosmoE10min(startValidTimeStr, endValidTimeStr, members = 'all', domainSize=512, outBaseDir='/scratch/ned/data/', cosmoBaseDir='/store/s83/tsm/EXP_TST/611/',rainThreshold=0.08,latencyTimeMin=100,lag=0,overwrite=False,useavailable=True):

    # datetime format
    startValidTime = ti.timestring2datetime(startValidTimeStr)
    endValidTime = ti.timestring2datetime(endValidTimeStr)
    timebounds = [startValidTime, endValidTime]
    
    # Check if the single nc file already exists
    if members=='all':
        year = startValidTime.year
        yearStr =  str(year)[2:4]
        julianDay = startValidTime.timetuple().tm_yday
        julianDayStr = '%03i' % julianDay
        yearJulianStr = yearStr + julianDayStr
        outDir = outBaseDir + startValidTime.strftime("%Y") + '/' + startValidTime.strftime("%y") + julianDayStr + '/'
        fcstName = 'COSMOE10min_' + startValidTime.strftime("%Y%m%d%H%M") + '_' + endValidTime.strftime("%Y%m%d%H%M") + '_lag' +  str(lag) + '_ltMin' + str(int(latencyTimeMin))
        fcstFile = r'%s' % (outDir + fcstName + '.nc')
    else:
        fcstFile = 'donotsavethisfile'
    
    # if not load original forecasts
    if (not os.path.isfile(fcstFile)) or overwrite:
        print(fcstFile + ' not found.')
        
        analysis_not_found = True
        nloops = 0
        while analysis_not_found and (nloops < 2):
        
            # Get most recent run (or second most recent)
            analysisTimeStr = rfe.find_nearest_forecast(startValidTimeStr, 'cosmo-e', lat_timeMin = latencyTimeMin, lag = lag)
            print('run time: ',analysisTimeStr)
            analysisTime = ti.timestring2datetime(analysisTimeStr)
            
            # list of EPS members to load
            if members == 'all':
                members = range(21)
                 
            # Folder to the nc files
            outDir = cosmoBaseDir + 'FCST' + analysisTime.strftime("%y") + '/' + analysisTime.strftime("%y%m%d%H") + '_611/output/'
            if os.path.isdir(outDir) == False:
                print('Folder not found: ' + outDir)
                if useavailable:
                    startValidTime -= datetime.timedelta(hours=12)
                    startValidTimeStr = ti.datetime2timestring(startValidTime)
                    nloops+=1
                else:
                    return None,None,None,None
            else:
                analysis_not_found=False

        if analysis_not_found:
            return None,None,None,None
        
        # Load individual EPS member and merge them in one array
        countm=0
        for member in members:
            thisFcstFile = outDir + 'cosmo-e_TOT_PREC_' + str(member).zfill(3) + '.nc'

            if not os.path.isfile(thisFcstFile):
                print('File not found: ' + thisFcstFile)
                sys.exit()
            else:
                print('Read: ' + thisFcstFile)
                
                # read the NetCDF file
                
                # this_member, timestamps, Xcoords, Ycoords = load_3darray_netcdf_with_bounds(thisFcstFile, timebounds, domainSize) # need to fix this...
                this_member, timestamps, Xcoords, Ycoords = load_3darray_netcdf(thisFcstFile)
                
                # print(timebounds)
                # print(this_member.shape)
                
                # flip and merge together the members
                if member == members[0]:
                    if domainSize>0:
                        cosmoe_data = np.zeros((this_member.shape[0], len(members), domainSize, domainSize))
                    else:
                        cosmoe_data = np.zeros((this_member.shape[0], len(members),this_member.shape[2],this_member.shape[3]))
                for i in xrange(this_member.shape[0]):
                    # flip frames
                    this_frame = np.flipud(this_member[i,0,:,:])
                    if domainSize>0:
                        cosmoe_data[i,countm,:,:] = dt.extract_middle_domain(this_frame, domainSize, domainSize)
                    else:
                        cosmoe_data[i,countm,:,:] = this_frame
                del this_member
            countm+=1
            
        # convert to mm/h
        cosmoe_data = cosmoe_data*6 
        
        # Get coordinates of reduced domain
        if domainSize>0:
            extent = dt.get_reduced_extent(Xcoords.shape[0], Ycoords.shape[0], domainSize, domainSize)
            Xmin = Xcoords[extent[0]]
            Ymin = Ycoords[extent[1]]
            Xmax = Xcoords[extent[2]]
            Ymax = Ycoords[extent[3]]    
            extent = (Xmin, Xmax, Ymin, Ymax)
            Xcoords = np.arange(Xmin,Xmax,1000)
            Ycoords = np.arange(Ymin,Ymax,1000) 
        else:
            extent = (Xcoords.min(), Xcoords.max(), Ycoords.min(), Ycoords.max())
        # Extract timestamps
        idxKeep1 = timestamps >= timebounds[0]
        idxKeep2 = timestamps <= timebounds[1]
        idxKeep = np.logical_and(idxKeep1,idxKeep2)
        cosmoe_data = cosmoe_data[idxKeep,:,:,:]
        timestamps = timestamps[idxKeep]
        
        # and store it before loading it again
        if (not fcstFile=='donotsavethisfile'):
            # print(fcstFile,cosmoe_data.shape,timestamps.shape,Xcoords.shape,Ycoords.shape)
            save_4darray_netcdf(fcstFile, cosmoe_data, 'COSMO-E10min',\
                            timestamps,cosmoe_data.shape[1],Xcoords,Ycoords)
    else:
        print('Read: ' + fcstFile)
        # Now read the NetCDF file
        cosmoe_data, timestamps, members, Ycoords, Xcoords = load_4darray_netcdf(fcstFile)
            
    # Apply rain threshold
    cosmoe_data[cosmoe_data<=rainThreshold] = 0
       
    testplot=False
    if testplot:
        nmember=0
        n=0
        while n<100:
            n+=1
            for t in xrange(timestamps.shape[0]):
                plt.clf()
                plt.imshow(10*np.log10(cosmoe_data[t,nmember,:,:]),interpolation ='nearest',vmin=-12,vmax=20, extent=[Xcoords.min(), Xcoords.max(), Ycoords.min(), Ycoords.max()])
                plt.title(timestamps[t])
                plt.pause(1)

    return cosmoe_data, timestamps, Xcoords, Ycoords   
    
def save_3darray_netcdf(fileName, dataArray, product,\
                        timestamps,xgrid,ygrid, \
                        noData=-99999.0):   

    # Get dimensions of output array to write
    nx = dataArray.shape[2]; ny = dataArray.shape[1]
    nt = dataArray.shape[0]
    
    # Set no data values
    dataArray[np.isnan(dataArray)] = noData
    
    # Make folder if necessary
    outDir = os.path.dirname(fileName)
    cmd = 'mkdir -p ' + outDir
    os.system(cmd)
              
    # Create netCDF Dataset
    w_nc_fid = netCDF4.Dataset(fileName, 'w', format='NETCDF4')
    w_nc_fid.Conventions =  'CF-1.6'
    w_nc_fid.ConventionsURL =  'http://www.unidata.ucar.edu/software/netcdf/conventions.html'
    w_nc_fid.institution = 'MeteoSwiss, Locarno-Monti'
    w_nc_fid.source = 'product: %s, product_category: determinist' % product
    w_nc_fid.history = 'Produced the ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Time dimension
    nrSamples = nt
    w_nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
    w_nc_time = w_nc_fid.createVariable('time', 'float', dimensions=('time'))
    w_nc_time.standard_name = 'time' 
    w_nc_time.long_name = 'time' 
    w_nc_time.units = 'seconds since %s' % timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
    w_nc_time.calendar = 'gregorian'
    w_nc_time[:] = netCDF4.date2num(timestamps, units = w_nc_time.units, calendar = w_nc_time.calendar)
    
    # Spatial dimension
    dimNames = ['x_1','y_1']
    dimensions = [int(nx),
                  int(ny)]
    for i in range(len(dimensions)):
        w_nc_fid.createDimension(dimNames[i],dimensions[i])
        
    # Write out coordinates
    w_nc_x = w_nc_fid.createVariable('x_1',np.dtype('float32').char,('x_1',))
    w_nc_x.axis = 'X' 
    w_nc_x.long_name = 'x-coordinate in Swiss coordinate system' 
    w_nc_x.standard_name = 'projection_x_coordinate' 
    w_nc_x.units = 'm' 
    w_nc_x[:] = xgrid
    
    w_nc_y = w_nc_fid.createVariable('y_1',np.dtype('float32').char,('y_1',))
    w_nc_y.axis = 'Y' 
    w_nc_y.long_name = 'y-coordinate in Swiss coordinate system' 
    w_nc_y.standard_name = 'projection_y_coordinate' 
    w_nc_y.units = 'm' 
    w_nc_y[:] = ygrid
    
    # Write out forecasts
    w_nc_PRECIP_INT = w_nc_fid.createVariable('PRECIP_INT', np.dtype('float32').char, dimensions=('time', 'y_1', 'x_1'), zlib=True, fill_value=noData)
    w_nc_PRECIP_INT.units = 'mm h-1'
    w_nc_PRECIP_INT.long_name = 'Precipitation intensity'
    w_nc_PRECIP_INT.coordinates = 'y_1 x_1'
    w_nc_PRECIP_INT[:] = dataArray
    
    w_nc_fid.close()
    print('Saved: ' + fileName)
 
def save_4darray_netcdf(fileName, dataArray, product,\
                        timestamps,nmebers,xgrid,ygrid, \
                        noData=-99999.0):   

    # Get dimensions of output array to write
    nx = dataArray.shape[3]; ny = dataArray.shape[2]
    nt = dataArray.shape[0]
    nm = dataArray.shape[1]
    
    # Set no data values
    dataArray[np.isnan(dataArray)] = noData
    
    # Make folder if necessary
    outDir = os.path.dirname(fileName)
    cmd = 'mkdir -p ' + outDir
    os.system(cmd)
              
    # Create netCDF Dataset
    w_nc_fid = netCDF4.Dataset(fileName, 'w', format='NETCDF4')
    w_nc_fid.Conventions =  'CF-1.6'
    w_nc_fid.ConventionsURL =  'http://www.unidata.ucar.edu/software/netcdf/conventions.html'
    w_nc_fid.institution = 'MeteoSwiss, Locarno-Monti'
    w_nc_fid.source = 'product: %s, product_category: probabilist' % product
    w_nc_fid.history = 'Produced the ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Time dimension
    nrSamples = nt
    w_nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
    w_nc_time = w_nc_fid.createVariable('time', 'float', dimensions=('time'))
    w_nc_time.standard_name = 'time' 
    w_nc_time.long_name = 'time' 
    w_nc_time.units = 'seconds since %s' % timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
    w_nc_time.calendar = 'gregorian'
    w_nc_time[:] = netCDF4.date2num(timestamps, units = w_nc_time.units, calendar = w_nc_time.calendar)
 
    # Member numbers
    nrMembers = nm
    w_nc_fid.createDimension('epsd_1', nrMembers) 
    w_nc_epsd = w_nc_fid.createVariable('epsd_1', 'int', dimensions=('epsd_1'))
    w_nc_epsd.standard_name = 'Epsd1' 
    w_nc_epsd.long_name = 'Ensemble member number' 
    w_nc_epsd.units = 'member number' 
    w_nc_epsd[:] = range(nrMembers)
    
    # Spatial dimension
    dimNames = ['x_1','y_1']
    dimensions = [int(nx),
                  int(ny)]
    for i in range(len(dimensions)):
        w_nc_fid.createDimension(dimNames[i],dimensions[i])
        
    # Write out coordinates
    w_nc_x = w_nc_fid.createVariable('x_1',np.dtype('float32').char,('x_1',))
    w_nc_x.axis = 'X' 
    w_nc_x.long_name = 'x-coordinate in Swiss coordinate system' 
    w_nc_x.standard_name = 'projection_x_coordinate' 
    w_nc_x.units = 'm' 
    w_nc_x[:] = xgrid
    
    w_nc_y = w_nc_fid.createVariable('y_1',np.dtype('float32').char,('y_1',))
    w_nc_y.axis = 'Y' 
    w_nc_y.long_name = 'y-coordinate in Swiss coordinate system' 
    w_nc_y.standard_name = 'projection_y_coordinate' 
    w_nc_y.units = 'm' 
    w_nc_y[:] = ygrid
    
    # Write out forecasts
    w_nc_PRECIP_INT = w_nc_fid.createVariable('PRECIP_INT', np.dtype('float32').char, dimensions=('time', 'epsd_1', 'y_1', 'x_1'), zlib=True, fill_value=noData)
    w_nc_PRECIP_INT.units = 'mm h-1'
    w_nc_PRECIP_INT.long_name = 'Precipitation intensity'
    w_nc_PRECIP_INT.coordinates = 'y_1 x_1'
    w_nc_PRECIP_INT[:] = dataArray
    
    w_nc_fid.close()
    print('Saved: ' + fileName)

def load_3darray_netcdf(filename):

    # read netcdf file
    nc_fid = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    variableNames = [str(var) for var in nc_fid.variables]

    # load time
    time_var = nc_fid.variables['time']
    timestamps = netCDF4.num2date(time_var[:],time_var.units)
    
    # load coordinates 
    x = nc_fid.variables["x_1"][:]
    y = nc_fid.variables["y_1"][:]
    Xcoords = np.array(x).squeeze()
    Ycoords = np.array(y).squeeze() 
         
    # load precip data
    data = nc_fid.variables[variableNames[-1]]
    noData = nc_fid.variables[variableNames[-1]]._FillValue 
    
    # convert to numpy array
    data = np.array(data)
    
    # change noData to Nan
    data[data==noData] = np.nan

    return data, timestamps, Xcoords, Ycoords 
    
def load_3darray_netcdf_with_bounds(filename, timebounds = [], domainSize = []):
    
    print('Error: this function needs to be fixed. Aborting.')
    sys.exit()
        
    # read netcdf file
    nc_fid = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    variableNames = [str(var) for var in nc_fid.variables]

    # load time
    time_var = nc_fid.variables['time']
    timestamps = netCDF4.num2date(time_var[:],time_var.units)
    
    # load coordinates 
    x = nc_fid.variables["x_1"][:]
    y = nc_fid.variables["y_1"][:]
    Xcoords = np.array(x).squeeze() - 500 # move coordinates to the lower left corner
    Ycoords = np.array(y).squeeze() - 500
    
    # prepare bounds 
    if len(timebounds) == 0:
        timebounds = [timestamps[0], timestamps[-1]]
        
    if isinstance(domainSize,int):
        extent = dt.get_reduced_extent(Xcoords.shape[0], Ycoords.shape[0], domainSize, domainSize)
        xbounds = [Xcoords[extent[0]], Xcoords[extent[2]]]
        ybounds = [Ycoords[extent[1]], Ycoords[extent[3]]]
    else:
        xbounds = [Xcoords[0], Xcoords[-1]]
        ybounds = [Ycoords[0], Ycoords[-1]]
         
    # time lower and upper index
    timeli = np.argmin( np.abs( timestamps - timebounds[0] ) )
    timeui = np.argmin( np.abs( timestamps - timebounds[1] ) ) + 1
    timestamps = timestamps[timeli:timeui]

    # x coords lower and upper index
    xli = np.argmin( np.abs( Xcoords - xbounds[0] ) )
    xui = np.argmin( np.abs( Xcoords - xbounds[1] ) )
    Xcoords = Xcoords[xli:xui]
    
    # y coords lower and upper index
    yli = np.argmin( np.abs( Ycoords - ybounds[0] ) )
    yui = np.argmin( np.abs( Ycoords - ybounds[1] ) )
    Ycoords = Ycoords[yli:yui]
    
    # load precip data
    data = nc_fid.variables[variableNames[-1]][timeli:timeui,:,xli:xui,yli:yui]
    noData = nc_fid.variables[variableNames[-1]]._FillValue 
    
    # convert to numpy array
    data = np.array(data)
    
    # change noData to Nan
    data[data==noData] = np.nan

    return data, timestamps, Xcoords, Ycoords 
    
def load_4darray_netcdf(filename):
    # COSMO-E

    # read netcdf file
    nc_fid = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    variableNames = [str(var) for var in nc_fid.variables]

    # load time
    time_var = nc_fid.variables['time']
    timestamps = netCDF4.num2date(time_var[:],time_var.units)

    # load precip
    data = nc_fid.variables[variableNames[-1]] 
    noData = data._FillValue 
    
    # convert to numpy array
    data = np.array(data)
    
    # change noData to Nan
    data[data==noData] = np.nan
    
    # load coordinates
    x = nc_fid.variables["x_1"][:]
    y = nc_fid.variables["y_1"][:]
    Xcoords = np.array(x).squeeze()
    Ycoords = np.array(y).squeeze()
    
    # load member numbers
    members_var = nc_fid.variables['epsd_1']
    members = np.array(members_var)

    return data, timestamps, members, Ycoords, Xcoords

def produce_forecast(timeStartStr,leadTimeMin):

    produce_radar_observations = False
    produce_radar_extrapolation = False
    produce_cosmo_1 = True
    
    # parameters
    rainThreshold = 0.08

    ## RADAR OBSERVATIONS
    if produce_radar_observations:
        print('Retrieve radar observations for ' + timeStartStr + ' + ' + str(leadTimeMin) + ' min')
        # start 5 min earlier to compute 10min accumulation at t0
        startTimeToPass = ti.timestring2datetime(timeStartStr) - datetime.timedelta(minutes=5)
        startTimeToPassStr = ti.datetime2timestring(startTimeToPass)
        radar_observations_5min, r = nw.get_radar_observations(startTimeToPassStr, leadTimeMin+5, product='RZC')
        # aggregate to 10-min forecast (to match COSMO1 resolution)
        radar_observations_10min = nw.aggregate_in_time(radar_observations_5min,timeAccumMin=10,type='sum')
        
    ## EXTRAPOLATION FORECAST
    if produce_radar_extrapolation:
        print('Run radar extrapolation for ' + timeStartStr + ' + ' + str(leadTimeMin) + ' min')
        # produce 5-min radar extrapolation
        radar_extrapolation_5min, radar_mask_5min = nw.radar_extrapolation(timeStartStr,leadTimeMin, product='RZC')
        # aggregate to 10-min forecast (to match COSMO1 resolution)
        radar_extrapolation_10min = nw.aggregate_in_time(radar_extrapolation_5min,timeAccumMin=10,type='sum')
        radar_mask_10min = nw.aggregate_in_time(radar_mask_5min,timeAccumMin=10,type='mean')
        # add observation at t0
        radar_extrapolation_10min = np.concatenate((radar_observations_10min[:,:,0,np.newaxis],radar_extrapolation_10min),axis=2)

    ## COSMO-1 FORECASTS
    if produce_cosmo_1:
        print('Retrive COSMO-1 for ' + timeStartStr + ' + ' + str(leadTimeMin) + ' min')
        cosmo1_10min = nw.get_cosmo1(timeStartStr, leadTimeMin)
        cosmo1_10min[cosmo1_10min <= rainThreshold] = np.nan
       
    # save_netcdf(radar_observations_10min,radar_extrapolation_10min,cosmo1_10min,\
                # timeStartStr,leadTimeMin,\
                # r.subXcoords,r.subYcoords)

def probability_matching(initialarray,targetarray):

    # zeros in initial image
    idxZeros = initialarray == 0

    # flatten the arrays
    arrayshape = initialarray.shape
    target = targetarray.flatten()
    array = initialarray.flatten()
    
    # rank target values
    order = target.argsort()
    ranked = target[order]

    # rank initial values order
    orderin = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[orderin] = np.arange(len(array))

    # get ranked values from target and rearrange with inital order
    outputarray = ranked[ranks]

    # reshape as 2D array
    outputarray = outputarray.reshape(arrayshape)
    
    # reassing original zeros
    outputarray[idxZeros] = 0

    return outputarray

# Probability matched mean
def build_PMM(ensemble):

    # Compute ensemble mean and build the rainrate PDF
    try: # list
        fieldShape = ensemble[0].shape
        nmembers = len(ensemble)
        ensemble_mean = np.zeros(fieldShape)
        members_per_pixel = np.zeros(fieldShape)
        for m in range(nmembers):
            memberField = ensemble[m].copy()
            idxNan = np.isnan(memberField)
            ensemble_mean[~idxNan] += memberField[~idxNan]
            members_per_pixel[~idxNan] += 1
            if m==0:
                rainrate_PDF = memberField.flatten()
            else:
                rainrate_PDF = np.concatenate((rainrate_PDF,memberField.flatten()))
    except TypeError: # numpy array
        fieldShape = ensemble[0,:,:].shape
        nmembers = ensemble.shape[0]
        ensemble_mean = np.zeros(fieldShape)
        members_per_pixel = np.zeros(fieldShape)
        for m in range(nmembers):
            memberField = ensemble[m,:,:].copy()
            idxNan = np.isnan(memberField)
            ensemble_mean[~idxNan] += memberField[~idxNan]
            members_per_pixel[~idxNan] += 1
            if m==0:
                rainrate_PDF = memberField.flatten()
            else:
                rainrate_PDF = np.concatenate((rainrate_PDF,memberField.flatten()))    
        
    ensemble_mean /= members_per_pixel   
    rainrate_PDF = np.sort(rainrate_PDF)[::-1]
    
    rainrate_PDF_nth = rainrate_PDF[::nmembers]
    
    pmm = probability_matching(ensemble_mean,rainrate_PDF_nth)

    return pmm
    
def load_n_random_radar_images(n,warThr = 10, yearStart = 2005, yearEnd = 2016, product='RZC'):
   
    images=[];timestamps=[]
    maxiter = n*50
    iter = 0
    naccepted = 0
    while (naccepted < n) and (iter < maxiter):
        iter+=1
        
        # Compute random date
        randomYear = int(np.random.uniform(yearStart,yearEnd+1,1))
        randomMonth = int(np.random.uniform(1,13,1))
        if randomMonth==12:
            daysMonth = 31
        else:
            daysMonth = (datetime.date(randomYear, randomMonth+1, 1) - datetime.date(randomYear, randomMonth, 1)).days
        randomDay = int(np.random.uniform(1,daysMonth+1))
        randomHour = int(np.random.uniform(0,24))
        randomMin = int(np.random.uniform(0,56))
        randomMin = int(5 * round(randomMin/5))
        randomDate = str(randomYear) + str(randomMonth).zfill(2) + str(randomDay).zfill(2) + str(randomHour).zfill(2) + str(randomMin).zfill(2)
        if product=='RZC':
            r = io.read_bin_image(randomDate, inBaseDir = '/scratch/ned/data/')
        else:
            r = io.read_gif_image(randomDate, inBaseDir = '/scratch/ned/data/') 
        if r.war > warThr:
            naccepted+=1
            images.append(r.rainrate)
            timestamps.append(randomDate)
            print(randomDate,int(r.war))

    return images,timestamps