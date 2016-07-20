import math
import numpy
import os
from osgeo import osr,gdal

RADIUS_E_WGS84  = 6378.137
RADIUS_P_WGS84  = 6356.7523142
spheroid = 'WGS84'
data_directory='/mnt/netapp/group/meradli/python/data'

def dms_to_dec(x):
  D = int(x[0:2])
  M = int(x[2:4])
  S = float(x[4:])
  DD = D + float(M)/60 + float(S)/3600
  return DD

def get_earth_radius(spheroid,latitude):
    """Get the radius of the Earth (in km) for a given Spheroid model at a givem position R^2 = ( a^4 cos(f)^2 + b^4 sin(f)^2 ) / ( a^2 cos(f)^2 + b^2 sin(f)^2 ).
    
    Arguments:     
        spheroid -  string defined spheroid (constant);

    Returns:

    """

    if spheroid == 'WGS84':
        RADIUS_E = RADIUS_E_WGS84
        RADIUS_P = RADIUS_P_WGS84
    latitude = math.radians(latitude)
    earth_radius = math.sqrt((math.pow(RADIUS_E,4) * math.pow(math.cos(latitude),2) + math.pow(RADIUS_P,4) * math.pow(math.sin(latitude),2) ) / ( math.pow(RADIUS_E,2) * math.pow(math.cos(latitude),2) +  math.pow(RADIUS_P,2) * math.pow(math.sin(latitude),2) ))
    return(earth_radius)

def sweep_edges(a1gate,nrays,rscale,nbins,elangle):
    """Compute sweep edges coordiantes."""
    a1gate = math.pi/2-a1gate
    azimuth = a1gate - numpy.linspace(0,2*math.pi,nrays + 1)
    coordinates = numpy.empty((nrays+1,nbins+1,3),dtype=float)
    coordinates[:,:,0] = numpy.tile(numpy.arange(nbins+1)*rscale ,(nrays + 1 ,1))
    coordinates[:,:,1] = numpy.transpose(numpy.tile(azimuth,(nbins + 1,1)))
    coordinates[:,:,2] = elangle
    return(coordinates)

def sweep_centers(a1gate,nrays,rscale,nbins,elangle):
    """Compute sweep centers coordiantes."""
    a1gate = math.pi/2-a1gate
    ascale = math.pi/nrays
    azimuth = a1gate + ascale/2 + numpy.linspace(0,2*math.pi,nrays,endpoint=False)
    coordinates = numpy.empty((nrays,nbins,3),dtype=float)
    coordinates[:,:,0] = numpy.tile(numpy.arange(nbins)*rscale + rscale/2,(nrays,1))
    coordinates[:,:,1] = numpy.transpose(numpy.tile(azimuth,(nbins,1)))
    coordinates[:,:,2] = elangle
    return(coordinates)

def ground_distance(height_radar, slant_range, elangle, earth_radius=6371250):
    """Calculate on ground distance and height of target point (Doviak, Zrnic).
    
    Arguments:     
        height_radar - height of the radar
        slant_range - distance from the radar
        elangle - elevation angle
        earth_radius - radius of the Earth in meters

    Returns:
        distance - distance at heigh_radar
        height - height above radar
    """
    radius = earth_radius*4/3;
    tmp1 = math.pow(radius + height_radar,2) + numpy.power(slant_range,2) + 2 * slant_range * (radius + height_radar) * math.sin(elangle);
    height = numpy.sqrt(tmp1) - radius - height_radar;
    distance = radius*numpy.arcsin(slant_range * math.cos(elangle) / (radius + height_radar + height))
    return [distance, height] 

def ground_distance2(height_radar, slant_range, elangle, earth_radius=6371250):
    """Calculate on ground distance and height of target point.
    
    Arguments:     
        height_radar - height of the radar
        slant_range - distance from the radar
        elangle - elevation angle
        earth_radius - radius of the Earth in meters

    Returns:
        distance - distance at heigh_radar
        height - height above radar
    """
    radius = earth_radius*4/3;
    tmp1 = slant_range * math.cos(elangle);
    tmp2 = slant_range * math.sin(elangle) + height_radar + radius;
    distance = radius * numpy.arctan(tmp1/tmp2);
    tmp1 = math.pow(radius + height_radar,2) + numpy.power(slant_range,2) + 2 * slant_range * (radius + height_radar) * math.sin(elangle);
    height = numpy.sqrt(tmp1) - radius - height_radar;
    return [distance, height] 

def get_cylindrical_coordinates_pvol(coord,height_radar,latitude_radar):
    """Get the cylindrical coordinates of the radar data.
    
    Arguments:     
        coord - 4D numpy array of spherical coordinates (slant range,alpha,phi)

    Returns:
        coordinates - 4D numpy array of cylindrical coordinates (slant range,alpha,phi)
    """

    ntilt,nrays,nbins,ncoord = coord.shape
    coordinates = numpy.empty(coord.shape,dtype=float)
    earth_radius = get_earth_radius(spheroid,latitude_radar);
    for n in range(0,ntilt):
        distance_ground, heigth_bin = ground_distance(height_radar,coord[n,0,:,0],coord[n,0,0,2],earth_radius*1000);
        coordinates[n,:,:,0] = numpy.tile(distance_ground,(nrays,1))
        coordinates[n,:,:,1] = coord[n,:,:,1]
        coordinates[n,:,:,2] = numpy.tile(heigth_bin,(nrays,1))
    return(coordinates)

def get_cylindrical_coordinates_sweep(coord,height_radar,latitude_radar):
    """Get the cylindrical coordinates of the radar data.
    
    Arguments:     
        coord - 3D numpy array of spherical coordinates (slant range,alpha,phi)

    Returns:
        coordinates - 3D numpy array of cylindrical coordinates (slant range,alpha,phi)
    """

    nrays,nbins,ncoord = coord.shape
    coordinates = numpy.empty(coord.shape,dtype=float)
    earth_radius = get_earth_radius(spheroid,latitude_radar);
    distance_ground, heigth_bin = ground_distance(height_radar,coord[0,:,0],coord[0,0,2],earth_radius*1000);
    coordinates[:,:,0] = numpy.tile(distance_ground,(nrays,1))
    coordinates[:,:,1] = coord[:,:,1]
    coordinates[:,:,2] = numpy.tile(heigth_bin,(nrays,1))
    return(coordinates)

def cylindrical_to_cartesian(coord):
    """Get the cartesian coordinates of the radar data.
    
    Arguments:     
        coord - 3D numpy array of cylindrical coordinates (range,alpha,z)

    Returns:
        coordinates - 3D numpy array of cartesian coordinates (x,y,z)
    """

    coordinates = numpy.empty(coord.shape,dtype=float)
    coordinates[...,0] = numpy.cos(coord[...,1])*coord[...,0]
    coordinates[...,1] = numpy.sin(coord[...,1])*coord[...,0]
    coordinates[...,2] = coord[...,2]
    return(coordinates)


def set_radar_projection(gdal_radar, latitude_radar, longitude_radar):
    """Set radar projection for the gdal object.
    
    Arguments:     

    Returns:
    """
    gdal_radar.SetProjection("+proj=aeqd  +lat_0=" + str(latitude_radar) + " +lon_0=" + str(longitude_radar) + " +x_0=0 + y_0=0")
    return(gdal_radar)

def get_geographical_coordinates(dataset):
    """Get the geographic coordinates from projected coordinates.
    
    Arguments:     
        dataset - gdal object (raster image with georeferencing)

    Returns:
        coordinates_geo - 3D numpy array of geographical coordinates (x,y,z)
    """

    coordinates_map = pixelToMap(dataset)
    coordinates_geo = projected_to_geographical(dataset.GetProjection(),coordinates_map)
    return(coordinates_geo)

def projected_to_geographical(proj4str,coordinates_map):
    """Get the geographic coordinates from projected coordinates.
    
    Arguments:     
        coordinates_map - 3D numpy array of projected coordinates (x,y,z)

    Returns:
        coordinates_geo - 3D numpy array of geographical coordinates (lat,lon,z)
    """

    ds_cs = get_proj4_projection(proj4str)
    geo_cs = get_default_projection()
    ct = osr.CoordinateTransformation(ds_cs,geo_cs)
    temp = coordinates_map.reshape((-1,3))
    coordinates_geo = numpy.array(ct.TransformPoints(temp));
    return(coordinates_geo.reshape(coordinates_map.shape))

def get_proj4_projection(proj4str):
    proj = osr.SpatialReference()
    proj.ImportFromProj4(proj4str)
    return proj

def get_default_projection():
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    return proj

def get_radar_projection(latitude_radar,longitude_radar):
    """Construct projection string of a weather radar"""
    projection = "+proj=aeqd  +lat_0=" + str(latitude_radar) + " +lon_0=" + str(longitude_radar) + " +x_0=0 + y_0=0"
    return(projection)

def geographical_to_projected(proj4str,coordinates_geo):
    """Get the projected coordinates from geographic coordinates.
    
    Arguments:     
        coordinates_geo - 3D numpy array of geographical coordinates (lat,lon,z)

    Returns:
        coordinates_map - 3D numpy array of projected coordinates (x,y,z)
    """

    ds_cs = get_proj4_projection(proj4str)
    geo_cs = get_default_projection()
    ct = osr.CoordinateTransformation(geo_cs, ds_cs)
    myshape = coordinates_geo.shape
    coordinates_geo = coordinates_geo.reshape(-1,myshape[-1])
    coordinates_map = numpy.array(ct.TransformPoints(coordinates_geo))
    return(coordinates_map.reshape(myshape[:-1] + (3,)))


def map_to_pixel(dataset,coordinates_map):
    """Computes pixel indices from projected coordinates given a gdal georeferenced object
    
    Arguments:     
        dataset - gdal object (raster image with georeferencing)
        coordinates_map - 3D numpy array of projected coordinates (x,y,z)

    Returns:
        coordinates_pixel - 3D numpy array of projected coordinates (x,y,z)
    """

    myshape = coordinates_map.shape
    coordinates_map = coordinates_map.reshape(-1,myshape[-1])
    geomatrix = dataset.GetGeoTransform()
    (success, inv_geo) = gdal.InvGeoTransform(geomatrix)
    (gx, gy) = applyGeoTransformVec(inv_geo, coordinates_map[:,0], coordinates_map[:,1])
    coordinates_pixel = numpy.array((gx, gy), dtype=int)
    return (coordinates_pixel.reshape(myshape))

def pixel_coordinates(dataset,mode="centers"):
    """Get coordinates of pixel centers from a georeferenced object.
    
    Arguments:     
     dataset -- gdal georeferenced object;

    Returns:
    coordinates -- 2D numpy array with coordinates

    """
    nx = dataset.RasterXSize
    ny = dataset.RasterYSize
    x = numpy.linspace(0,nx,num=nx)
    y = numpy.linspace(0,ny,num=ny)
    if  mode == "centers":
        x = x + 0.5
        y = y + 0.5
        numpy.delete(x,-1)
        numpy.delete(y,-1)
    X,Y = numpy.meshgrid(x,y)
    coordinates = numpy.empty(X.shape + (2,))
    coordinates[:,:,0] = X
    coordinates[:,:,1] = Y
    return (coordinates)

def pixel_to_map(dataset,coordinates=None,mode="centers"):
    """Get projected coordinates from pixel coordinates from a georeferenced object.
    
    Arguments:     
     dataset -- gdal georeferenced object;

    Returns:
    coordinates_map -- 2D numpy array with coordinates

    """

    if (coordinates == None):
        coordinates = pixel_coordinates(dataset,mode)
    geomatrix = dataset.GetGeoTransform()
    coordinates_map = numpy.empty(coordinates.shape[:-1] + (3,))
    coordinates_map[...,0:2] = apply_geotransform_vector(geomatrix, coordinates)
    coordinates_map[...,2] = numpy.ones(coordinates.shape[:-1])
    return (coordinates_map)

def apply_geotransform_vector(geomatrix,coordinates):
    """Get coordinates of pixel centers from a georeferenced object.
    
    Arguments:     
     geomatrix -- transformation matrix;
    coordinates -- 2D numpy array with pixel coordinates

    Returns:
    coordinates -- 2D numpy array with projected coordinates

    """
    coordinates_map = numpy.empty(coordinates.shape)
    coordinates_map[...,0] = geomatrix[0] + geomatrix[1] * coordinates[...,0] + geomatrix[2] * coordinates[...,1]
    coordinates_map[...,1] = geomatrix[3] + geomatrix[4] * coordinates[...,0] + geomatrix[5] * coordinates[...,1]
    return(coordinates_map)


