#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

import time_tools_attractor as ti
ti.tic()
import getpass
usrName = getpass.getuser()

import argparse
import sys
import os
import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab
from pylab import get_cmap

from sklearn.externals import joblib # same purpose as pickle but more efficient with big data / can only pickle to disk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# My modules
import gis_base as gis
import data_tools_attractor as dt
import time_tools_attractor as ti

ti.toc('to load python modules.')

fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"

#########################################################################
############# PARAMETERS FOR SYNTHETIC CASES ############################
X_predictors = ['x_d', 'y_d', 'u', 'v', 'hzt_o'] #, 'daytime_sin', 'daytime_cos']

## Parameters for synthetic cases
sectorSizeDeg = 5

flowDirection_degN_array = np.arange(0,360, sectorSizeDeg)
#flowDirection_degN_array = [180+45]

flowSpeed_kmh_array = [30]

hztHeight_m_array = np.arange(500,4600,100)
hztHeight_m_array = [3000]

dayTime_array = np.arange(0,24)
dayTime_array = [12]

#########################################################################

################# INPUT ARGS ############################################
parser = argparse.ArgumentParser(description='Compute MAPLE archive statistics.')
parser.add_argument('-model', default='mlp', type=str, help='Which model to train [mlp, ols, dt] or the file containing the already trained model to use for predictions.')
parser.add_argument('-q', default=None, type=float, nargs='+', help='Which quantile to predict. Only available with qrf (quantile random forests). If two values are passed it rpedicts the difference.')
parser.add_argument('-fmt', default='png', type=str, help='Figure format.')
args = parser.parse_args()

####
tmpDir = '/scratch/lforesti/ml_tmp/'
outBaseDir = '/users/lforesti/results/maple/ml/'
fig_fmt = args.fmt
if fig_fmt == 'png':
    fig_dpi = 100
elif (fig_fmt == 'pdf') or (fig_fmt == 'eps'):
    fig_dpi = None
else:
    fig_dpi = 200

# Check input model
model_list = ['mlp', 'ols', 'dt', 'gp', 'knn', 'qrf']
if (args.model in model_list):
    print('Train', args.model, 'model.')
elif os.path.isfile(args.model):
    print('Load', args.model)
else:
    print('Invalid -model option. Either train a', model_list, 'model or load an already trained mdoel file.')
    sys.exit()
print('')

########################################################################################################
###### LOAD already trained model ######################################################################
if os.path.isfile(args.model):
    print('No training.')
    best_model = joblib.load(args.model)
    print(args.model, 'read.')
    print(best_model)
    
    if hasattr(best_model, 'variables'):
        X_predictors = best_model.variables        
else:
    print('File', args.model, 'not found.')
    sys.exit()
        
############### SELECT PREDICTORS ######################################################################
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

########################################################################################################
####### PREDICT ON NEW SYNTHETIC DATA ##################################################################
# Animate PNGs
# convert mlp_predictions_flow_225SW_speed30kmh_hzt*m.png -set delay 40 -duplicate 1,-2-1 anim_SW.gif

import maple_dataload
cmaps = maple_dataload.generate_colormaps()
geo = maple_dataload.generate_geo()

# Generate grid of spatial coordinates (extent of predictions)
res_km = 8
x_min = 310
x_max = 910
y_min = -100
y_max = 440

extent_image = np.array([x_min-res_km/2, x_max+res_km/2, y_min-res_km/2, y_max+res_km/2])*1000
print(extent_image)

x_vec = np.arange(x_min, x_max + res_km, res_km)
y_vec = np.arange(y_min, y_max + res_km, res_km)
x_grid, y_grid = np.meshgrid(x_vec, y_vec)
y_grid = np.flipud(y_grid)
                    
# Generate array of flow directions
flowDirection_compassShort_array = dt.deg2compass(flowDirection_degN_array, 'short')
flowDirection_compassLong_array = dt.deg2compass(flowDirection_degN_array, 'long')

print('Predicting with', best_model.name.upper(), 'on synthetic cases...')
for flowDirection_degN, flowDirection_compassShort, flowDirection_compassLong  in zip(flowDirection_degN_array, flowDirection_compassShort_array, flowDirection_compassLong_array):
    for flowSpeed_kmh in flowSpeed_kmh_array:
        for hztHeight_m in hztHeight_m_array:
            for dayTime in dayTime_array:

                # Generate homogeneous grids of flow vectors and hzt
                flowDirection_deg = dt.degN2deg(flowDirection_degN)
                flowDirection_rad = np.deg2rad(flowDirection_deg)
                u = -flowSpeed_kmh*np.cos(flowDirection_rad) # Remember to turn the vectors!
                v = -flowSpeed_kmh*np.sin(flowDirection_rad)
                #print(flowDirection_compassShort, flowDirection_degN, flowDirection_deg, u, v)
                
                u_grid = u*np.ones(x_grid.shape)
                v_grid = v*np.ones(x_grid.shape)
                hzt_grid = hztHeight_m*np.ones(x_grid.shape)
                
                # Generate homogeneous grids of daytime (sin,cos)
                dayTimeSin, dayTimeCos = ti.daytime2circular(dayTime)
                dayTimeSin_grid = dayTimeSin*np.ones(x_grid.shape)
                dayTimeCos_grid = dayTimeCos*np.ones(x_grid.shape)
                
                # Flatten
                X_pred = np.column_stack((x_grid.flatten(), y_grid.flatten(), u_grid.flatten(), v_grid.flatten(), hzt_grid.flatten(), dayTimeSin_grid.flatten(), dayTimeCos_grid.flatten()))
                X_pred_scaled = scaler.transform(X_pred)
                    
                # Select predictors
                X_pred_scaled = X_pred_scaled[:, X_ids]
                
                #################
                # Predict growth and decay on grid
                y_pred = best_model.predict(X_pred_scaled)
                cmap = cmaps.cmapLog
                norm = cmaps.normLog
                clevs = cmaps.clevsLog
                
                # Reshape results
                y_pred_grid = np.reshape(y_pred, x_grid.shape)

                ########### PLOTTING ##########################
                # Plot predictions
                ratioFigWH = (x_max-x_min)/(y_max-y_min)
                figWidth = 10
                fig = plt.figure(figsize=(figWidth,figWidth/ratioFigWH))
                ax = fig.add_subplot(111)

                # Draw DEM
                ax.imshow(geo.demImg, extent=geo.extent_CCS4, cmap=cmaps.cmapDEM, norm=cmaps.normDEM, alpha=cmaps.alphaDEM)
                # Draw prediction map
                im = plt.imshow(y_pred_grid, extent=extent_image, cmap=cmap, norm=norm, interpolation='nearest')#, alpha=cmaps.alpha)
                # Draw contour 1000 m
                ax.contour(np.flipud(geo.demImg_smooth), levels=[1000], colors='gray', extent=extent_image, alpha=0.5)
                # Draw mask
                ax.imshow(geo.radarMask, cmap=cmaps.cmapMask, extent=geo.radarExtent, alpha=0.5)
                # Axes
                plt.xticks([], [])
                plt.yticks([], [])
                xRange = (geo.extent_smalldomain[1] - geo.extent_smalldomain[0])/1000
                yRange = (geo.extent_smalldomain[3] - geo.extent_smalldomain[2])/1000
                ax.set_xlabel(str(int(xRange)) + ' km', fontsize=18, y=-0.025)
                ax.set_ylabel(str(int(yRange)) + ' km', fontsize=18, x=-0.025)
                # Extent reduced domain
                plt.xlim([geo.extent_smalldomain[0], geo.extent_smalldomain[1]])
                plt.ylim([geo.extent_smalldomain[2], geo.extent_smalldomain[3]])
                # # Title all in one-line
                # titleStr = 'MLP predictions of MAP growth and decay, ' + str(flowDirection_degN).zfill(3) + r'$^\circ$' + flowDirection_compassShort.ljust(3,' ') + ' flows, HZT ' + str(hztHeight_m) + ' m'
                # plt.title(titleStr, fontsize=15, loc='left')
                
                # Draw arrow
                x_arrow_a = [720, 540, 700] #480
                y_arrow_a = [150, 200, 250] #280
                if (len(hztHeight_m_array) == 1) and (len(flowDirection_degN_array) > 1):
                    for x_arrow, y_arrow in zip(x_arrow_a, y_arrow_a):
                        plt.quiver(x_arrow*1000, y_arrow*1000, u*1000, v*1000, units='xy', pivot='middle', width=2000)
                if (len(hztHeight_m_array) > 1) and (len(flowDirection_degN_array) == 1):
                    for x_arrow, y_arrow in zip(x_arrow_a, y_arrow_a):
                        plt.text(x_arrow*1000, y_arrow*1000, str(hztHeight_m) + ' m', fontsize=16, horizontalalignment='center')
                
                # Title on two lines      
                titleStr1 = best_model.name.upper() + ' predictions of MAP growth and decay ' + '\n'
                titleStr2 = str(flowDirection_degN).rjust(3,' ') + r'$^\circ$' + flowDirection_compassShort.ljust(3,' ') + ' flows, HZT ' + str(hztHeight_m) + ' m'
                plt.title(titleStr1 + titleStr2, fontsize=15)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(im, cax=cax, ticks=clevs, spacing='uniform', norm=norm, extend='both')
                cbTitle = '   dB'
                clb.ax.set_title(cbTitle, fontsize=18)
                clb.ax.tick_params(labelsize=17)
                # Draw shapefile
                gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=ax, linewidth=0.75, alpha=0.5)
                # Draw radars, cities, regions
                dt.draw_radars(ax, which=['LEM','DOL','ALB','PPM'], fontsize=10, marker='^', markersize=50, markercolor='w', only_location=True)
                #dt.draw_cities(ax, fontsize=10, marker='o', markersize=8, markercolor='k')    
                dt.draw_regions(ax, fontsize=11, color='k')
                # Text input predictors
                x_txt = 0.01
                y_txt = 1
                line_s = 0.035
                ftsize_txt = 13
                alignment = 'left'
                # plt.text(x_txt, y_txt, 'Input predictors:', transform=ax.transAxes, fontsize=ftsize_txt, horizontalalignment=alignment)
                # plt.text(x_txt, y_txt-line_s, 'Flow direction = ' + flowDirection_compassLong, transform=ax.transAxes, fontsize=ftsize_txt, horizontalalignment=alignment)
                plt.text(x_txt, y_txt-1*line_s, 'Flow speed = ' + str(flowSpeed_kmh) + r' km h$^{-1}$', transform=ax.transAxes, fontsize=ftsize_txt, horizontalalignment=alignment)
                # plt.text(x_txt, y_txt-3*line_s, 'Freezing elvel height = ' + str(hztHeight_m) + ' m', transform=ax.transAxes, fontsize=ftsize_txt, horizontalalignment=alignment)
                plt.text(1.0, y_txt-1*line_s, "%02i" % dayTime + ' UTC', transform=ax.transAxes, fontsize=ftsize_txt, horizontalalignment='right')
                
                ### Filename
                if ('u' in X_predictors) and ('v' in X_predictors):
                    flowString = '_flow_' + str(flowDirection_degN).zfill(3) + flowDirection_compassShort + '_speed' +  str(flowSpeed_kmh) + 'kmh_hzt'
                else:
                    flowString = ''
                    
                if ('hzt_o' in X_predictors) or ('hzt_d' in X_predictors):
                    hztString = '_' + str(hztHeight_m).zfill(4) + 'm' 
                else:
                    hztString = ''
                    
                if ('daytime_sin' in X_predictors) and ('daytime_cos' in X_predictors):
                    daytimeString = '_' + ("%02i" % dayTime) + 'UTC'
                else:
                    daytimeString = ''
                
                fileNameFig = outBaseDir + best_model.name + '/' + best_model.name + '_predictions'  + flowString  + hztString + daytimeString + '.' + fig_fmt
                
                fig.savefig(fileNameFig, dpi=fig_dpi, bbox_inches='tight')
                print(fileNameFig, 'saved.')
                plt.close()

print('./maple_machine-learning_predict-maps.py finished!')