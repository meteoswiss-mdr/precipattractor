#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib as mpl
mpl.use('Agg')

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gis_base as gis

import numpy as np                       
import matplotlib.pyplot as plt
import matplotlib

import stat_tools_attractor as st
import data_tools_attractor as dt
import io_tools_attractor as io

import maple_dataload
geo = maple_dataload.generate_geo()

def colorbar(mappable,r0,label=r'   mm h$^{-1}$',labelsize=13,ticklabelsize=10):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=r0.clevs, spacing='uniform', norm=r0.norm, extend='max', orientation='vertical')
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.set_ticklabels(r0.clevsStr, update_ticks=True)
    cax.text(1,0.5,clabel,fontsize=labelsize,transform=cax.transAxes,ha='left',va='top',rotation='vertical')

def plot_frames_paper(startStr,t,r0,data,f,g,minR=0.01,plotmember=1,figsize=(16,5),labelsize=14):

    plt.close()
    
    # extract radar masks
    nanmask = f.getmask
    alphav = 0.5

    # load DEM and Swiss borders
    shp_path = "/users/ned/pyscripts/shapefiles/CHE_adm0.shp"
    proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84"
    proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
    +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 
    dem_path = '/users/ned/pyscripts/shapefiles/ccs4.png'
    dem = Image.open(dem_path)
    dem = dt.extract_middle_domain_img(dem, 512, 512)
    dem = dem.convert('P')
    
    figmatrix = [1,3]

    fig, axes = plt.subplots(nrows=figmatrix[0], ncols=figmatrix[1], figsize=figsize)
    
    # OBS
    a = axes[0]
    fplot = data.obs[t,0,:,:]
    fplot[fplot<=minR]=np.nan
    obsmask = np.ones(fplot.shape)
    obsmask[data.mask==1] = np.nan
    a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
    obsIm = a.imshow(fplot,extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
    a.imshow(obsmask,extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
    plt.text(0,.99,'radar QPE', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top')
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])    
    gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
    a.text(1,0,data.timestamps[t].strftime('%Y-%m-%d %H:%M' ), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
    # NOWCAST f
    a = axes[2]
    if plotmember<0: # ensemble mean
        fplot = np.nanmean(f.getnowcast,axis=0)
        idxmask = 0
    else:
        fplot = f.getnowcast[plotmember,:,:].copy()
        idxmask =  plotmember
    fplot[fplot<=minR]=np.nan
    a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
    bayIm = a.imshow(fplot,extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
    # a.imshow(nanmask[idxmask,:,:],extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
    plt.text(0.01,.99,'+ growth/decay', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top')
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])    
    gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
    a.text(1,0,'+%i min' % (t*10), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
    # NOWCAST g
    a = axes[1]
    if plotmember<0: # ensemble mean
        fplot = np.nanmean(g.getnowcast,axis=0)
        idxmask = 0
    else:
        fplot = g.getnowcast[plotmember,:,:].copy()
        idxmask =  plotmember
    fplot[fplot<=minR]=np.nan
    a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
    bayIm = a.imshow(fplot,extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
    # a.imshow(nanmask[idxmask,:,:],extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
    plt.text(0.01,.99,'just extrapolation', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top')
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])    
    gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
    a.text(1,0,'+%i min' % (t*10), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.93)
    cax = fig.add_axes([0.95, 0.08, 0.011, 0.86])
    cbar = fig.colorbar(bayIm, cax=cax, ticks=r0.clevs, spacing='uniform', norm=r0.norm, extend='max')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticklabels(r0.clevsStr, update_ticks=True)
    a.text(1.04,0.5,'mm h$^{-1}$',fontsize=14,transform=axes[2].transAxes,ha='left',va='center',rotation='vertical')

    figname = 'tmp/fig_frame_' + startStr + '_' + str(t).zfill(2) + '.png'
    fig.savefig(figname)
    print('saved: ' + figname)    
 