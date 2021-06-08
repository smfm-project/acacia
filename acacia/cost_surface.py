# Use a cost surface to produce a map of travel time from points in sub-Saharan Africa

import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import scipy.ndimage
import shapefile

import biota

import pdb


# Build a travel time map around a single point in a country


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int(round((x - ulX) / xDist))
    line = int(round((ulY - y) / xDist))
    return (pixel, line)


def buildFrictionSurface(shp, city, coord, friction_surface = '/home/sbowers3/DATA/acacia/DATA/friction_surface_2015_v1.0.tif', step = 0.5):
    """
    """
    
    print 'Doing %s'%city
        
    output_dir = '/home/sbowers3/DATA/acacia/outputs/%s'%city
    
    command = 'gdalwarp -cutline %s -crop_to_cutline -q -overwrite %s %s/%s'%(shp, friction_surface, output_dir, friction_surface.split('/')[-1][:-4] + '_%s.tif'%city)
    
    os.system(command)                              
    
    ds = gdal.Open('%s/%s'%(output_dir, friction_surface.split('/')[-1][:-4] + '_%s.tif'%city), 0)
    friction_surface = ds.ReadAsArray()#[12000:14000,25000:26000] # Units: mins/m
    
    # Convert surface to mins / pixel
    friction_surface = friction_surface * ds.GetGeoTransform()[1] * 110000. # Convert to mins / pixel (approx)
    friction_surface[friction_surface == 0] = 999999.
    
    # Round to nearest step size
    friction_surface = np.round(friction_surface * (1./step), 0) / (1. / step)
    
    arrived = np.zeros_like(friction_surface).astype(np.bool)
    
    x, y = world2Pixel(ds.GetGeoTransform(), coord[1], coord[0])
    arrived[y,x] = True
    
    time_elapsed = np.zeros_like(friction_surface).astype(np.float32)
    arrival_time = np.zeros_like(friction_surface).astype(np.float32)
    
    for mins in np.arange(0, 1440 + 1, step):
        
        if mins % 100 == 0: print mins
        
        adjacent_px = np.logical_and(scipy.ndimage.morphology.binary_dilation(arrived, scipy.ndimage.generate_binary_structure(2, 2)), arrived == False)
        
        time_elapsed[adjacent_px] += step
        
        sel = np.logical_and(time_elapsed > friction_surface, adjacent_px)
        
        arrived[sel] = True
        
        arrival_time[sel] = mins


    arrival_time[arrival_time == 0] = 999999.
    
    output_name = '%s/travel_time_%s.tif'%(output_dir,city)
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(output_name, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(ds.GetGeoTransform())
    ds_out.SetProjection(ds.GetProjection())
    ds_out.GetRasterBand(1).SetNoDataValue(999999.)
    ds_out.GetRasterBand(1).WriteArray(arrival_time)
    ds_out = None
    
    return output_name
    

if __name__ == '__main__':
    """
    """
    
    cities = ['DarEsSalaam', 'Lusaka', 'Maputo', 'Luanda', 'Lilongwe', 'Lubumbashi', 'Harare', 'Nairobi']
    shps = ['TZA_adm0.shp', 'ZMB_adm0.shp', 'MOZ_adm0.shp', 'AGO_adm0.shp', 'MWI_adm0.shp', 'COD_adm0.shp', 'ZWE_adm0.shp', 'KEN_adm0.shp']
    coords = [[-6.7924, 39.2083], [-15.3875, 28.3228], [-25.9692, 32.5732], [-8.8147, 13.2302], [-13.9626, 33.7741], [-11.6876, 27.5026], [-17.8252, 31.0335], [-1.2921, 36.8219]]
    
    shps = ['/home/sbowers3/DATA/acacia/DATA/shapefiles/' + shp for shp in shps]
    
    output_name = buildFrictionSurface(shps[-2], cities[-2], coords[-2])
    output_name = buildFrictionSurface(shps[-1], cities[-1], coords[-1])
    
    #for city, shp, coord in zip(cities, shps, coords):
    #    output_name = buildFrictionSurface(shp, city, coord)
    


    ### DarEsSalaam settings
    #shp = '/home/sbowers3/DATA/acacia/DATA/shapefiles/TZA_adm0.shp'
    #city = 'DarEsSalaam'
    #y, x = 694, 1193
    
    ### Lusaka settings
    #shp = '/home/sbowers3/DATA/acacia/DATA/shapefiles/ZMB_adm0.shp'
    #city = 'Lusaka'
    #y, x = 863, 756
    
    
    ### Maputo settings
    #shp = '/home/sbowers3/DATA/acacia/DATA/shapefiles/MOZ_adm0.shp'
    #city = 'Maputo'
    #y, x = 1855, 285

