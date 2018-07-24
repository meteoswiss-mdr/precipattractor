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
cmaps = maple_dataload.generate_colormaps()

def colorbar(mappable,r0,label=r'   mm h$^{-1}$',labelsize=13,ticklabelsize=10):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=r0.clevs, spacing='uniform', norm=r0.norm, extend='max', orientation='vertical')
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.set_ticklabels(r0.clevsStr, update_ticks=True)
    cax.text(1,0.5,clabel,fontsize=labelsize,transform=cax.transAxes,ha='left',va='top',rotation='vertical')

def set_axis_off(a):
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])
    
def plot_frames_paper(startStr,t,r0,data,f,g,minR=0.01, plotmember=[0], gd_field=None, figsize=(16,5),labelsize=14):

    plt.close()
    
    # extract radar masks
    nanmask = f.getmask
    alphav = 0.5

    proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84"
    proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
    +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 
    dem_path = '/store/mch/msrad/radar/precip_attractor/gis_data/dem/ccs4.png'
    dem = Image.open(dem_path)
    dem = dt.extract_middle_domain_img(dem, 512, 512)
    dem = dem.convert('P')
    
    nMembers = len(plotmember)
    numberPlots = 2*nMembers+1 # nwc with gd, without gd, observations, gd field
    if gd_field is not None:
        numberPlots+=1
    
    figmatrix = dt.optimal_size_subplot(numberPlots)
    nrRows = figmatrix[0]
    nrCols = figmatrix[1]
    ratioWidthHeight = nrCols/nrRows
    if nrRows == nrCols:
        figsize = (11, 9.5)
    else:
        figsize = (13, 13/ratioWidthHeight)
    
    # figmatrix = [1,3]
    fig,axes = plt.subplots(nrows=nrRows, ncols=nrCols, figsize=figsize)
    axes_stack = []
    
    # OBS
    a = plt.subplot(nrRows, nrCols, 1)
    axes_stack.append(a)
    fplot = data.obs[t,0,:,:]
    fplot[fplot<=minR]=np.nan
    obsmask = np.ones(fplot.shape)
    obsmask[data.mask==1] = np.nan
    a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
    obsIm = a.imshow(fplot,extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
    a.imshow(obsmask,extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
    plt.text(0,.99,'radar QPE', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top')
    set_axis_off(a)
    gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
    a.text(1,0,data.timestamps[t].strftime('%Y-%m-%d %H:%M' ), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
    # FX
    for m in range(0, nMembers):
        # NOWCAST without GD
        a = plt.subplot(nrRows, nrCols, 2+m*nMembers)
        axes_stack.append(a)
        if plotmember[m]<0: # ensemble mean
            fplot = np.nanmean(f.getnowcast, axis=0)
            idxmask = 0
        else:
            fplot = f.getnowcast[plotmember[m],:,:].copy()
            idxmask =  plotmember[m]
        fplot[fplot<=minR]=np.nan
        a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
        bayIm = a.imshow(fplot, extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
        # a.imshow(nanmask[idxmask,:,:],extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
        plt.text(0.01,.99,'+ growth/decay', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top', color='r')
        if plotmember[m]<0:
            plt.text(0.99,.99,'ensemble mean', fontsize=labelsize,transform=a.transAxes, ha = 'right', va='top')
        else:
            plt.text(0.99,.99,'member ' + str(m), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='top')
        set_axis_off(a)
        gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
        a.text(1,0,'+%i min' % (t*10), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
        # NOWCAST with GD
        a = plt.subplot(nrRows, nrCols, 3+m*nMembers)
        axes_stack.append(a)
        if plotmember[m]<0: # ensemble mean
            fplot = np.nanmean(g.getnowcast,axis=0)
            idxmask = 0
        else:
            fplot = g.getnowcast[plotmember[m],:,:].copy()
            idxmask =  plotmember[m]
        fplot[fplot<=minR]=np.nan
        a.imshow(dem, extent=r0.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'), alpha=0.8)
        bayIm = a.imshow(fplot,extent = r0.extent, interpolation='nearest', cmap=r0.cmap, norm=r0.norm)
        # a.imshow(nanmask[idxmask,:,:],extent=r0.extent,interpolation='nearest',cmap=r0.cmapMask,alpha=alphav)
        plt.text(0.01,.99,'just extrapolation', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top', color='b')
        if plotmember[m]<0:
            plt.text(0.99,.99,'ensemble mean', fontsize=labelsize,transform=a.transAxes, ha = 'right', va='top')
        else:
            plt.text(0.99,.99,'member ' + str(m), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='top')
        set_axis_off(a)
        gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
        a.text(1,0,'+%i min' % (t*10), fontsize=labelsize,transform=a.transAxes, ha = 'right', va='bottom')
    
    # Plot GD
    if gd_field is not None:
        a = plt.subplot(nrRows, nrCols, 2+m+nMembers+1)
        axes_stack.append(a)
        gdIm = plt.imshow(gd_field, extent=r0.extent, cmap=cmaps.cmapLog, norm=cmaps.normLog, interpolation='nearest')
        gis.read_plot_shapefile(geo.fileNameShapefile, geo.proj4stringCH, geo.proj4stringCH, ax=a, linewidth=0.75, alpha=0.5)
        set_axis_off(a)
        plt.text(0.01,.99,'growth/decay field', fontsize=labelsize,transform=a.transAxes, ha = 'left', va='top')
        # # Colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # clb = fig.colorbar(im, cax=cax, ticks=clevs, spacing='uniform', norm=norm, extend='both')
        # cbTitle = '   dB'
        # clb.ax.set_title(cbTitle, fontsize=18)
        # clb.ax.tick_params(labelsize=17)
    
    plt.tight_layout()
    
    # Rainfall colormap
    if numberPlots <= 4:
        xpos = 0.95
    else:
        xpos = 0.93
    fig.subplots_adjust(right=xpos)
    cax = fig.add_axes([xpos, 0.52, 0.015, 0.45])   
    cbar = fig.colorbar(bayIm, cax=cax, ticks=r0.clevs, spacing='uniform', norm=r0.norm, extend='max')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticklabels(r0.clevsStr, update_ticks=True)
    cbar.ax.set_title('mm h$^{-1}$',fontsize=11) #transform=axes[0, nrCols-1].transAxes
    
    # GD colormap
    fig.subplots_adjust(right=xpos)
    cax = fig.add_axes([xpos, 0.02, 0.015, 0.45])
    cbar = fig.colorbar(gdIm, cax=cax, cmap=cmaps.cmapLog, ticks=cmaps.clevsLog, norm=cmaps.normLog, extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_title('dB',fontsize=11)
    
    figname = 'tmp/fig_frame_' + startStr + '_' + str(t).zfill(2) + '.png'
    
    return(figname, axes_stack)
 
