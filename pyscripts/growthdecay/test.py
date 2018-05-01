#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib as mpl
mpl.use('Agg')

import numpy as np                       
import matplotlib.pyplot as plt

import cv2
import argparse

from scipy import fftpack

import nowcasting as nw
import io_tools_attractor as io
import data_tools_attractor as dt
import stat_tools_attractor as st
import time_tools_attractor as ti
import ssft


# advection schemes
import maple_ree

from nowcast import Datamanager, Nowcasting
from plotting_functions import plot_frames_paper

from sklearn.externals import joblib # same purpose as pickle but more efficient with big data / can only pickle to disk
from sklearn import preprocessing

# Parameters
np.random.seed(1)

#+++++++++++++++++++++++++++++++++++ Get arguments from command line

parser = argparse.ArgumentParser(description='Test nowcasting.')
parser.add_argument('-start', default='201701311600', type=str, help='Starting date YYYYMMDDHHmm.')
parser.add_argument('-end', default='201701312200', type=str, help='Ending date YYYYMMDDHHmm.')

args = parser.parse_args()

startStr = args.start[0:12]
endStr = args.end[0:12]

start = ti.timestring2datetime(startStr)

#+++++++++++++++++++++++++++++++++++ Parameters

leadtime_hrs = 4
timestep_res_min = 10
domain_size = 512
upscale_km = 1                 
N = 21
AR_order = 2                  
number_levels = 8
min_rainrate = 0.08
cosmo_model = 'COSMO-E'
transformation = 'dBR'          # 'dBR', 'dBZ', False
probability_matching = True     # probability matching
advection_scheme = 'maple'      # 'maple','euler'
zero_padding = 32

maxleadtime_min = leadtime_hrs*60
nts = int(maxleadtime_min/timestep_res_min)
    

#+++++++++++++++++++++++++++++++++++ Load the data

data = Datamanager(startStr,endStr,upscale_km,min_rainrate=min_rainrate)

r0 = io.read_bin_image('201604161900',inBaseDir='/scratch/ned/data/', minR=min_rainrate, fftDomainSize=domain_size) # for plotting parameters only

#+++++++++++++++++++++++++++++++++++ Compute motion field
radar_nlast = data.n_last_obs(3)
radarStack=[]
minDBZ = 10.0*np.log10(316*min_rainrate**1.5)
minDBR = 10.0*np.log10(min_rainrate)
for i in xrange(3):
    arrayin = radar_nlast[i,:,:]
    if (transformation=='dBZ'):
        arrayin,_,_ = dt.rainrate2reflectivity(arrayin, 316.0, 1.5, minDBZ)
        arrayin = arrayin - minDBZ
    if (transformation=='dBR'):
        idxwet = arrayin>0
        arrayout = arrayin.copy()
        arrayout[idxwet] = dt.to_dB(arrayin[idxwet])
        arrayout[~idxwet] = minDBR
        arrayout = arrayout - minDBR
    radarStack.append(arrayin)
nrOfFields = len(radarStack)
U,V = nw.get_motion_field(radarStack, doplot=0, verbose=1, resKm = upscale_km, resMin = timestep_res_min)

#+++++++++++++++++++++++++++++++++++ Predict growth&decay

model = '/scratch/lforesti/ml_tmp/mlp_predictors/mlp_model_7-100-1_xyuv-hzt-utc.pkl'
tmpDir = '/scratch/lforesti/ml_tmp/'

if os.path.isfile(model):
    print('No training.')
    best_model = joblib.load(model)
    print(model, 'read.')
    print(best_model)
    
    if hasattr(best_model, 'inputs'):
        X_predictors = best_model.inputs  
else:
    print('File', model, 'not found.')
    sys.exit()
    
# List of all variable names
X_varnames = ['x_d', 'y_d', 'u', 'v', 'hzt_o', 'daytime_sin', 'daytime_cos']
X_varnames_dict = dict(zip(X_varnames, np.arange(0,len(X_varnames))))

print('Predictors chosen:')
print(X_predictors)
X_ids = [X_varnames_dict[k] for k in X_predictors]

##### Load file to scale the data
fileName_scaler = tmpDir + 'scaler.pkl'
scaler = joblib.load(fileName_scaler)    
print(fileName_scaler, 'read.')

# Generate grid of spatial coordinates (extent of predictions)
res_km = upscale_km
x_min = 310
x_max = 910
y_min = -100
y_max = 440

extent_image = np.array([x_min-res_km/2, x_max+res_km/2, y_min-res_km/2, y_max+res_km/2])*1000
print(extent_image)

# x, y
x_vec = np.arange(x_min, x_max + res_km, res_km)
y_vec = np.arange(y_min, y_max + res_km, res_km)
x_grid, y_grid = np.meshgrid(x_vec, y_vec)
y_grid = np.flipud(y_grid)
x_grid = dt.extract_middle_domain(x_grid, int(domain_size/res_km), int(domain_size/res_km))
y_grid = dt.extract_middle_domain(y_grid, int(domain_size/res_km), int(domain_size/res_km))

# U and V vectors
u_grid = U*60/timestep_res_min*upscale_km
v_grid = -V*60/timestep_res_min*upscale_km

# HZT
hztHeight_m = 1000
hzt_grid = hztHeight_m*np.ones(x_grid.shape)

# Daytime
y_pred_grid_disaggr = np.zeros((x_grid.shape[0],x_grid.shape[1],int(leadtime_hrs*60/timestep_res_min)))
for lt in xrange(leadtime_hrs):
    dayTime = start.hour + lt
    dayTimeSin, dayTimeCos = ti.daytime2circular(dayTime)
    dayTimeSin_grid = dayTimeSin*np.ones(x_grid.shape)
    dayTimeCos_grid = dayTimeCos*np.ones(x_grid.shape)

    # Flatten predictors
    X_pred = np.column_stack((x_grid.flatten(), y_grid.flatten(), u_grid.flatten(), v_grid.flatten(), hzt_grid.flatten(), dayTimeSin_grid.flatten(), dayTimeCos_grid.flatten()))
    X_pred_scaled = scaler.transform(X_pred)
    
    # Select predictors
    X_pred_scaled = X_pred_scaled[:, X_ids]

    # Predict growth and decay on grid 
    y_pred = best_model.predict(X_pred_scaled) # !! hourly prediction

    # Reshape results
    y_pred_grid = np.reshape(y_pred, x_grid.shape)

    plt.close()
    plt.imshow(y_pred_grid,vmin=-2,vmax=2,interpolation='none',cmap=plt.get_cmap('coolwarm'))
    plt.colorbar()
    plt.savefig('growthdecay-%i.png' % lt)
    
    # Disaggregate growth&decay
    y_pred_grid_disaggr[:,:,int(lt*60/timestep_res_min):int((lt+1)*60/timestep_res_min)] = nw.compute_advection(y_pred_grid,-U,-V,net=60/timestep_res_min)[:,:,::-1]/(60/timestep_res_min)
    
#+++++++++++++++++++++++++++++++++++ Compute parameters for AR(2) process

# (0 build hanning window
window = ssft.build2dWindow((int(domain_size/upscale_km),int(domain_size/upscale_km)),'flat-hanning')

# (1) advect all observations to t0
radarStack_at_t0 = []
for n in np.arange(0,nrOfFields-1):
    tmp = nw.compute_advection(radarStack[n].copy(),U,V,net=nrOfFields-n-1)[:,:,-1] # maple
    radarStack_at_t0.append(tmp) 
radarStack_at_t0.append(radarStack[-1].copy())

# (2) cascade decomposition of the last radar observations
fftshape = tuple(x+zero_padding*2 for x in radarStack[0].shape) 
bandpassFilter2D,centreWaveLengths = nw.calculate_bandpass_filter(fftshape, number_levels, Width = 2.0, resKm = upscale_km, doplot=True)
cascadeStack, cascadeMeanStack, cascadeStdStack = nw.get_cascade_from_stack(radarStack_at_t0*window, number_levels, bandpassFilter2D, centreWaveLengths, zeroPadding = zero_padding, doplot=1)

# (3) estimate phi
phi,r = nw.autoregressive_parameters(cascadeStack, cascadeMeanStack, cascadeStdStack, AR_order)
print('Phi:')
print('\n'.join('{}: {}'.format(*k) for k in enumerate(phi)))

#+++++++++++++++++++++++++++++++++++ Define extrapolation model

def fx(x, rf=[1,1], net=1):
    xin = x.copy()
    # resize motion fields by factor f (for advection)
    fr = 0.5
    if (fr<1):
        Ures = cv2.resize(U, (0,0), fx=fr, fy=fr)
        Vres = cv2.resize(V, (0,0), fx=fr, fy=fr) 
    else:
        Ures = U
        Vres = V
    return maple_ree.ree_epol_slio(xin, Vres*rf[0], Ures*rf[1], net)[:,:,-1]

#+++++++++++++++++++++++++++++++++++ Define random noise function

domsize = int(domain_size/upscale_km)
freq = fftpack.fftfreq(domsize,d=1.0)
ref = radarStack[-1].copy()
filter = np.fft.fft2(ref*window)
filter = np.abs(filter)
def rx(shape,n=1):
    randValues = np.random.randn(n,shape[0],shape[1])
    corrNoise = np.zeros_like(randValues)
    for nn in xrange(n):
        fnoise = np.fft.fft2(randValues[nn,:,:])
        fcorrNoise = fnoise*filter
        corrNoise[nn,:,:] = np.array(np.fft.ifft2(fcorrNoise).real)
        corrNoise[nn,:,:] = ( corrNoise[nn,:,:] - corrNoise[nn,:,:].mean() ) / corrNoise[nn,:,:].std()
    
    return corrNoise
    
#+++++++++++++++++++++++++++++++++++ Define upscaling function

def ux(fieldin, upscale_km, resKm = 1, wavelet = 'haar'):

    # wavelets decomposition
    wavelet_coeff = st.wavelet_decomposition_2d(fieldin, wavelet, nrLevels = None)
    
    # Generate coordinates of centers of wavelet coefficients
    extent = (0, fieldin.shape[1], 0, fieldin.shape[0])
    xvecs, yvecs = st.generate_wavelet_coordinates(wavelet_coeff, fieldin.shape, extent[0], extent[1], extent[2], extent[3], resKm)

    # Append a given wavelet scale to write out into daily netCDF files
    scale2keep = st.get_level_from_scale(resKm, upscale_km)

    scaleKm = int(xvecs[scale2keep][1] - xvecs[scale2keep][0])
    if upscale_km != scaleKm:
        print('Asked and returned wavelet scales not matching.', upscale_km, 'vs', scaleKm)
        sys.exit()
    
    return wavelet_coeff[scale2keep]
        
##########################################
####### NOWCASTING #######################
##########################################


#+++++++++++++++++++++++++++++++++++ Init the nowcast 

f = Nowcasting(data, N, hx=[], fx, rx, phi, AR_order=AR_order, number_levels=number_levels, transformation=transformation, probability_matching=probability_matching, resolution_km=upscale_km, label='with growthdecay', min_rainrate=min_rainrate, zero_padding=zero_padding)

g = Nowcasting(data, N, hx=[], fx, rx, phi, AR_order=AR_order, number_levels=number_levels, transformation=transformation, probability_matching=probability_matching, resolution_km=upscale_km, label='without growthdecay', min_rainrate=min_rainrate, zero_padding=zero_padding)

#+++++++++++++++++++++++++++++++++++ Start nowcasting

for t in xrange(nts):

    print('t = %i' % t)
    plot_frames_paper(startStr,t,r0,data,f,g,min_rainrate)
    
    #+++++++++++++++++++++++++++++++++++ Predict  --> t+1
    f.predict(growthdecay = y_pred_grid_disaggr[:,:,t]) 
    g.predict(growthdecay = None) 
    