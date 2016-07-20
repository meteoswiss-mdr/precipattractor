"""
Module to perform basic GIS operations on vectors (shapefiles) and rasters.

"""
import shapefile
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np
from osgeo import gdal, osr, ogr
import os
import geo

def get_proj4_ogr(filename):
    src = ogr.Open(filename, 0)
    layer = src.GetLayerByIndex(0)
    srs = layer.GetSpatialRef()
    if srs == None:
        return("+init=epsg:31370")
        #return("+init=epsg:3812")
    src = None
    return(srs.ExportToProj4())

def read_plot_shapefile(fileName, s_srs = None, t_srs = None,  ax = None, linewidth = .5):
    """
    Function to read, project and plot a shapefile.
    Arguments:
    - fileName - file name of shapefile
    - s_srs - PROJ4 string of the shapefile projection (source projection)
    - t_srs - PROJ4 string of the plotting projection (destination projection)
    Returns:
    - ax - axes handle of the plot (used for overlaying). The axes can be passed as well to the function.
    """
    ############### EXTENT ###################### 
    if s_srs is None:
        s_srs =  get_proj4_ogr(fileName)
    s_srs = geo.get_proj4_projection(s_srs)
    t_srs = geo.get_proj4_projection(t_srs)
    ct = osr.CoordinateTransformation(s_srs,t_srs)
    sf = shapefile.Reader(fileName)
    recs = sf.records()
    shapes = sf.shapes()
    Nshp = len(shapes)
    extent = shapes[0].bbox
    xminM = extent[0]
    xmaxM = extent[2]
    yminM = extent[1]
    ymaxM = extent[3]
    for nshp in xrange(Nshp):
        extent = shapes[nshp].bbox
        xmin = extent[0]
        if xmin < xminM:
                xminM = xmin
        xmax = extent[2]
        if xmax > xmaxM:
                xmaxM = xmax
        ymin = extent[1]
        if ymin < yminM:
                yminM = ymin
        ymax = extent[3]
        if ymax > ymaxM:
                ymaxM = ymax

    extent = (xminM,  yminM,  xmaxM,  ymaxM)
    
    ################ PLOTTING ######################
    cns     = []
    path = []
    if ax is None: # to define before the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for nshp in xrange(Nshp): # for every polygon
        cns.append(recs[nshp][4])
        cm = plt.get_cmap('Dark2')
        cccol = cm(1.*np.arange(Nshp)/Nshp)
        ptchs   = []
        temp = np.array(shapes[nshp].points)
        n = temp.shape[0]
        temp2 = np.zeros((n,3))
        temp2[:,0:2] = temp
        if (t_srs !=  None):
            pts = np.array(ct.TransformPoints(temp2))
            pts = pts[:,0:2]
            prt = shapes[nshp].parts
            par = list(prt) + [pts.shape[0]]
        for pij in xrange(len(prt)): # for every point in the polygon
            ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
            ax.add_collection(PatchCollection(ptchs,facecolor='none',edgecolor='k', linewidths=linewidth))
    return(ax)

def gdal_merge_raster(outputFileName,  *inputfileNames):
    """
    Function to merge different raster images into a single image.
    Arguments:
    - outputFileName - filename of the output raster
    - inputfileNames - filenames of input rasters to merge (separated by comma as different function arguments)
    """
    # open first file to get no data value
    rst = gdal.Open(inputfileNames[0])
    b1 = rst.GetRasterBand(1)
    nodata = b1.GetNoDataValue()
    bandtype = gdal.GetDataTypeName(b1.DataType)    

    if (os.path.isfile(outputFileName) == True):
        os.system("rm " + outputFileName)
    cmd = 'gdalwarp -ot ' + bandtype + ' -srcnodata ' + str(nodata) + ' -dstnodata ' + str(nodata)
    for arg in inputfileNames:
        cmd += " " + arg
    
    cmd += " " + outputFileName
    print cmd
    os.system(cmd)

def gdal_merge_resample_raster_res(xres,  yres,  outputFileName,  *inputfileNames):
    """
    Function to merge and resample different raster images into a single image.
    Arguments:
    - xres - x resolution
    - yres - yresolution
    - outputFileName - filename of the output raster
    - inputfileNames - filenames of input rasters to merge (separated by comma as different function arguments)
    """
    # open first file to get no data value
    rst = gdal.Open(inputfileNames[0])
    b1 = rst.GetRasterBand(1)
    nodata = b1.GetNoDataValue()
    bandtype = gdal.GetDataTypeName(b1.DataType)    

    if (os.path.isfile(outputFileName) == True):
        os.system("rm " + outputFileName)
    cmd = 'gdalwarp -ot ' + bandtype + ' -tr ' + str(xres) + " " + str(yres) + ' -srcnodata ' + str(nodata) + ' -dstnodata ' + str(nodata)
    for arg in inputfileNames:
        cmd += " " + arg
    
    cmd += " " + outputFileName
    print cmd
    os.system(cmd)

def gdal_merge_resample_raster(nrRows,  nrCols,  outputFileName,  *inputfileNames):
    """
    Function to merge and resample different raster images into a single image.
    Arguments:
    - nrRows - number of rows of the output images
    - nrCols - number of columns of the output image
    - outputFileName - filename of the output raster
    - inputfileNames - filenames of input rasters to merge (separated by comma as different function arguments)
    """
    # open first file to get no data value
    rst = gdal.Open(inputfileNames[0])
    b1 = rst.GetRasterBand(1)
    nodata = b1.GetNoDataValue()
    bandtype = gdal.GetDataTypeName(b1.DataType)    

    if (os.path.isfile(outputFileName) == True):
        os.system("rm " + outputFileName)
    
    cmd = 'gdalwarp -ot ' + bandtype + ' -ts ' + str(nrRows) + " " + str(nrCols) + ' -srcnodata ' + str(nodata) + ' -dstnodata ' + str(nodata)
    for arg in inputfileNames:
        cmd += " " + arg
    
    cmd += " " + outputFileName
    print cmd
    os.system(cmd)


def gdal_project_raster(inputFileName,  outputFileName,  proj4stringSource, proj4stringTarget):
    """
    Function to project a raster
    """
    rst = gdal.Open(inputFileName)
    b1 = rst.GetRasterBand(1)
    nodata = b1.GetNoDataValue()
    bandtype = gdal.GetDataTypeName(b1.DataType)    
    
    if (os.path.isfile(outputFileName) == True):
        os.system("rm " + outputFileName)

    cmd = 'gdalwarp -s_srs \"%s\" -t_srs \"%s\" -dstnodata %i %s %s' % (proj4stringSource, proj4stringTarget , nodata, inputFileName,  outputFileName)
    cmd = 'gdalwarp -ot ' + bandtype + ' -s_srs \"' + proj4stringSource + '\" -t_srs \"' + proj4stringTarget + '\" -srcnodata ' + str(nodata) + ' -dstnodata ' + str(nodata) + ' ' + inputFileName + ' ' + outputFileName
    print cmd    
    os.system(cmd)

def gdal_clip_raster(inputFileName, outputFileName, extent):
    """
    Function to clip a raster. Extent = [ulx, uly, lrx, lry]
    """
    cmd = 'gdal_translate -projwin ' + str(extent[0]) + ' ' + str(extent[1]) + ' ' + str(extent[2]) + ' ' + str(extent[3]) + ' ' + inputFileName + ' ' + outputFileName
    print cmd
    os.system(cmd)

def gdal_simplify_vector(outputFileName, inputFileName, tolerance):
    """
    Function to simplify the border of the polygons of a shapefile
    Arguments:
    - tolerance - length level of simplification (given in input coordinates)
    """
    cmd = 'ogr2ogr %s %s -simplify %f -overwrite' % (outputFileName, inputFileName, tolerance)    
    os.system(cmd)

def gdal_merge_vector(outputFileName, *inputFileNames):
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileNames[0] + ' -overwrite'
    print cmd
    os.system(cmd)
    i = 0
    for arg in inputFileNames:
        if (i!=0):
            cmd = 'ogr2ogr -update -append ' + outputFileName + ' ' + arg
            print cmd
            os.system(cmd)
        i=i+1
    #cmd = 'ogrinfo ' + outputFileName + ' -sql \"CREATE SPATIAL INDEX ON ' + 'merged' + '\"'
    #os.system(cmd)
    #print cmd

def gdal_merge_vector_list(outputFileName, inputFileNames):
    """
    Same as gdal_merg_vector but passing the shapefile names as a list
    """
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileNames[0] + ' -overwrite'
    print cmd
    os.system(cmd)
    for i in range(1,len(inputFileNames)):
        cmd = 'ogr2ogr -update -append ' + outputFileName + ' ' + inputFileNames[i] + ' -f \"esri shapefile\"'# -nln merged'
        print cmd
        os.system(cmd)
    #cmd = 'ogrinfo -sql \"CREATE SPATIAL INDEX ON merged\" ' + outputFileName
    #print cmd
    #os.system(cmd)
    #print cmd
    #ogr2ogr -update -append merged.shp file2.shp -nln merge

def gdal_clip_vector(inputFileName, outputFileName, xmin, xmax, ymin, ymax):
    """
    Function to clip a shapefile. Clipping consists of selecting a rectangular region and cutting away the rest.
    """
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileName + ' -overwrite'
    print cmd
    os.system(cmd)
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileName + ' -f \"ESRI Shapefile\" -clipdst ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + ' -overwrite'
    print cmd
    os.system(cmd)

def gdal_project_shapefile(inputFileName, outputFileName, s_srs = None, t_srs = None):
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileName + ' -overwrite'
    print cmd
    os.system(cmd)
    cmd = 'ogr2ogr ' + outputFileName + ' ' + inputFileName + ' -f \"ESRI Shapefile\" -s_srs \"' + s_srs + '\" -t_srs \"' + t_srs + '\" -overwrite' 
    print cmd
    os.system(cmd)

def gdal_read_raster(fileName):
    """
    Function to read a raster image.
    Returns:
    lon - longitude coordinates. Generated given the image reference
    lat - latitude coordinates. Generated given the image reference
    image - raster image
    """
    rst = gdal.Open(fileName)
    b1 = rst.GetRasterBand(1)
    nodata = b1.GetNoDataValue()
    image = b1.ReadAsArray()
    #image[image==nodata] = 'NaN'

    ## create projected coordinates image
    cols = rst.RasterXSize
    rows = rst.RasterYSize
    orig_reference = rst.GetGeoTransform()
    orig_proj = rst.GetProjection()

    lon_pixels = np.linspace(0.5, cols-0.5, cols)
    lat_pixels = np.linspace(0.5, rows-0.5, rows)
    ULLon = orig_reference[0]
    ULLat = orig_reference[3]
    lon = ULLon + orig_reference[1]*lon_pixels + orig_reference[2]*lon_pixels
    lat  = ULLat +orig_reference[4]*lat_pixels + orig_reference[5]*lat_pixels
    return(lon, lat,  image)

def set_shade(a,vmin,vmax,intensity=None,cmap=cm.jet,scale=10.0,azdeg=165.0,altdeg=45.0):
    ''' sets shading for data array based on intensity layer
    or the data's value itself.
    inputs:
    a - a 2-d array or masked array
    intensity - a 2-d array of same size as a (no chack on that)
            representing the intensity layer. if none is given
            the data itself is used after getting the hillshade values
            see hillshade for more details.
    cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
          instance)
    scale,azdeg,altdeg - parameters for hilshade function see there for
          more details
    output:
    rgb - an rgb set of the Pegtop soft light composition of the data and 
       intensity can be used as input for imshow()
    based on ImageMagick's Pegtop_light:
    http://www.imagemagick.org/Usage/compose/#pegtoplight'''
    if intensity is None:
        # hilshading the data
        intensity = hillshade(a,scale=10.0,azdeg=165.0,altdeg=45.0)
    else:
        # or normalize the intensity
        intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    # get rgb of normalized data based on cmap
    rgb = cmap((a-vmin)/float(vmax-vmin))[:,:,:3]
    # form an rgb eqvivalent of intensity
    d = intensity.repeat(3).reshape(rgb.shape)
    # simulate illumination based on pegtop algorithm.
    rgb = 2*d*rgb+(rgb**2)*(1-2*d)
    return rgb

def hillshade(data,scale=10.0,azdeg=165.0,altdeg=45.0):
    ''' convert data to hillshade based on matplotlib.colors.LightSource class.
    input:
     data - a 2-d array of data
     scale - scaling value of the data. higher number = lower gradient
     azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
              270 west
     altdeg - where the light comes from: 0 horison ; 90 zenith
    output: a 2-d array of normalized hilshade
    '''
    # convert alt, az to radians
    az = azdeg*pi/180.0
    alt = altdeg*pi/180.0
    # gradient in x and y directions
    dx, dy = gradient(data/float(scale))
    slope = 0.5*pi - arctan(hypot(dx, dy))
    aspect = arctan2(dx, dy)
    intensity = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(-az - aspect - 0.5*pi)
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity

def gdal_hillshade(inputFileName, outputFileName, zFactor = 1, azDegrees=315):
    '''
    Computes the hillshade from a digital elevation model using GDAL library.
    zFactor: vertical height exaggeration
    azDegrees: location of the virtual light source

    '''
    if (os.path.isfile(outputFileName) == True):
        os.system("rm " + outputFileName)
    
    cmd = 'gdaldem hillshade ' + inputFileName + ' ' +  outputFileName + ' -z ' + str(zFactor) + ' -s 1 -az ' +  str(azDegrees)
    os.system(cmd)
