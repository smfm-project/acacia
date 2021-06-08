# Map change within city catchment

import csv
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal

import biota
import biota.IO

import pdb




def doTile(lat, lon, t1, t2, city, data_dir = '/home/sbowers3/SMFM/ALOS_data/', max_travel_time = 720, uid_min = 0):
    """
    max_travel_time: Fruthest reaches of city influence in minutes, defaults to 720 mins (12 hours)
    """
    
    print 'Doing %s'%city
    
    travel_time = '/home/sbowers3/DATA/acacia/outputs/%s/travel_time_%s.tif'%(city, city)
    output_dir = '/home/sbowers3/DATA/acacia/outputs/%s/'%city
    
    # Load city region
    ds = gdal.Open(travel_time, 0)

    geo_t = ds.GetGeoTransform()

    xmin = np.int(np.floor(geo_t[0]))
    xmax = np.int(np.ceil(geo_t[0] + ds.RasterXSize * geo_t[1]))
    ymax = np.int(np.ceil(geo_t[3]))
    ymin = np.int(np.floor(geo_t[3] + ds.RasterYSize * geo_t[5]))
    
    AGB_availability = np.zeros((max_travel_time+1))
    AGB_area = np.zeros((max_travel_time+1))
        
    print 'Doing lat: %s, lon: %s'%(str(lat), str(lon))
                    
    try:
        tile_t1 = biota.LoadTile(data_dir, lat, lon, t1, lee_filter = True, window_size = 3, contiguity = 'rook', output_dir = output_dir)
        tile_t2 = biota.LoadTile(data_dir, lat, lon, t2, lee_filter = True, window_size = 3, contiguity = 'rook', output_dir = output_dir)
    except:
        print '    No data, continuing...'
        return uid_min
    
    travel_time_tile = biota.IO.loadRaster(travel_time, tile_t1, resampling = gdal.GRA_Bilinear, dtype = gdal.GDT_Float32)
    
    # Set city catchment to 12 hours travel time
    tt_mask = np.logical_and(travel_time_tile <= max_travel_time, travel_time_tile > 0)
    
    if tt_mask.sum() == 0:
        print '    Outside of mask, continuing...'
        return uid_min
    
    tile_t1.mask = np.logical_or(tile_t1.mask, tt_mask == False)
    tile_t2.mask = np.logical_or(tile_t2.mask, tt_mask == False)
    
    tile_t1.updateMask('/home/sbowers3/guasha/sam_bowers/carla_outputs/DATA/hansen_water/hansen_water_mask.vrt', buffer_size = 1000., classes = [2])
    tile_t2.updateMask('/home/sbowers3/guasha/sam_bowers/carla_outputs/DATA/hansen_water/hansen_water_mask.vrt', buffer_size = 1000., classes = [2])
    
    AGB_t1 = tile_t1.getAGB(output = True)
    
    # Sum AGB per minute travel_time
    #travel_time_tile_round = np.round(travel_time_tile).astype(np.int)
    #for mins in np.unique(np.round(travel_time_tile).astype(np.int)):
    #    if mins == 0 or mins > max_travel_time: continue
    #    AGB_availability[mins] += np.nansum(AGB_t1[travel_time_tile_round == mins]) * (tile_t1.xRes * tile_t1.yRes * 0.0001)
    #    AGB_area[mins] += np.nansum(travel_time_tile_round == mins) * (tile_t1.xRes * tile_t1.yRes * 0.0001)
        
    tile_change = biota.LoadChange(tile_t1, tile_t2, change_area_threshold = 1, change_intensity_threshold = 0.2, change_magnitude_threshold = 5., contiguity = 'rook', output_dir = output_dir)
    
    AGB_change = tile_change.getAGBChange(output = True)
    Change_class = tile_change.getChangeType(output = True)
    
    # Get a unique ID for each each polygon
    _, location_id = biota.indices.getContiguousAreas(np.logical_or(Change_class['deforestation'], Change_class['degradation']), True, contiguity = 'rook')
    
    location_id[location_id > 0] = location_id[location_id > 0] + uid_min
    
    # Output ChangeID
    biota.IO.outputGeoTiff(location_id.astype(np.int32), tile_change.output_pattern%'ChangeID', tile_change.geo_t, tile_change.proj, output_dir = tile_change.output_dir, dtype = gdal.GDT_Int32, nodata = 999999)
    
    # Output travel time
    travel_time_tile[tt_mask == False] = 9999
    biota.IO.outputGeoTiff(np.round(travel_time_tile).astype(np.int), tile_change.output_pattern%'TravelTime', tile_change.geo_t, tile_change.proj, output_dir = tile_change.output_dir, dtype = gdal.GDT_Int16, nodata = 9999)

    
    #with open(output_dir + 'AGB_availability_%s.csv'%city, 'wb') as csvfile:
    #    writer = csv.writer(csvfile, delimiter = ',')
    #    writer.writerow(['Travel time (mins)', 'AGB availability (tC)', 'Area (ha)'])
    #    for mins, AGB, area in zip(np.arange(max_travel_time+1).tolist(), AGB_availability.tolist(), AGB_area.tolist()):
    #        writer.writerow([str(mins), str(AGB), str(area)])
    
    # Get max UID, except where no change events exist return uid_min
    if location_id[np.logical_and(location_id > 0, location_id != 999999)].shape[0] > 0:
        uid_max = (location_id[np.logical_and(location_id > 0, location_id != 999999)]).max()
    else:
        uid_max = uid_min.copy()
    
    return uid_max


if __name__ == '__main__':
    """
    """
    
    cities = ['DarEsSalaam', 'Lusaka', 'Maputo', 'Luanda', 'Lilongwe', 'Lubumbashi', 'Harare']#, 'Nairobi']
    #cities = ['Maputo']
    #cities = ['DarEsSalaam']
    
    t1 = 2007
    t2 = 2016
    
    last_val = 0
    
    for city in cities:

        travel_time = '/home/sbowers3/DATA/acacia/outputs/%s/travel_time_%s.tif'%(city, city)
        
        # Load city region
        ds = gdal.Open(travel_time, 0)

        geo_t = ds.GetGeoTransform()

        xmin = np.int(np.floor(geo_t[0]))
        xmax = np.int(np.ceil(geo_t[0] + ds.RasterXSize * geo_t[1]))
        ymax = np.int(np.ceil(geo_t[3]))
        ymin = np.int(np.floor(geo_t[3] + ds.RasterYSize * geo_t[5]))
                
        for lat in range(ymin, ymax)[::-1]:
            for lon in range(xmin, xmax):
                last_val = doTile(lat, lon, 2007, 2016, city, uid_min = last_val)
                #last_val += max_uid
