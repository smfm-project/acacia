# Calculate class probability by distance, AGB availability etc.

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pandas as pd

# Get AGB from t1 and t2


def getTotalAGBChangeFromGeotiff(input_dir):
    '''
    Not yet taking to account change
    '''
    
    AGB_t1_files = sorted(glob.glob('%s/AGB_2007_*.tif'%input_dir))
    AGBChange_files = sorted(glob.glob('%s/AGBChange_2007_2016_*.tif'%input_dir))
    tt_files = sorted(glob.glob('%s/TravelTime_2007_2016_*.tif'%input_dir))
    ChangeClass_files = sorted(glob.glob('%s/ChangeClass_2007_2016_*.tif'%input_dir))

    assert len(AGB_t1_files) == len(AGBChange_files) == len(tt_files) == len(ChangeClass_files), "Inputs must be of the same length"
        
    AGB_t1_total = np.zeros((721), dtype = np.float64)
    AGB_t2_total = np.zeros((721), dtype = np.float64)
    AGB_area = np.zeros((721), dtype = np.float64)
    AGB_ChangeClass_total = np.zeros((721, 6), dtype = np.float64)
    AGB_ChangeClass_area = np.zeros((721, 6), dtype = np.float64)
    
    for AGB_t1_file, AGBChange_file, tt_file, ChangeClass_file in zip(AGB_t1_files, AGBChange_files, tt_files, ChangeClass_files):
        
        print 'Doing %s'%AGB_t1_file.split('/')[-1]
        
        AGB_t1 = gdal.Open(AGB_t1_file, 0).ReadAsArray()
        AGB_change = gdal.Open(AGBChange_file).ReadAsArray()  
        tt = gdal.Open(tt_file, 0).ReadAsArray()
        ChangeClass = gdal.Open(ChangeClass_file, 0).ReadAsArray()
                
        mask = np.logical_or(np.logical_or(AGB_t1 == 999999., AGB_change == 999999), tt == 9999) == False
         
        AGB_t2 = np.zeros_like(AGB_t1) + 999999.
        AGB_t2[mask] = AGB_t1[mask] + AGB_change[mask]
        
        AGB_t1_total += scipy.stats.binned_statistic(tt[mask].flatten(), AGB_t1[mask].flatten(), bins=np.arange(0,722), statistic='sum')[0] * (25 * 25 * 0.0001)
        AGB_t2_total += scipy.stats.binned_statistic(tt[mask].flatten(), AGB_t2[mask].flatten(), bins=np.arange(0,722), statistic='sum')[0] * (25 * 25 * 0.0001)
        AGB_area += scipy.stats.binned_statistic(tt[mask].flatten(), np.ones_like(AGB_t1[mask].flatten()), bins=np.arange(0,722), statistic='sum')[0] * (25 * 25 * 0.0001)
        
        for cc in range(1,7):
            cc_mask = np.logical_and(mask, ChangeClass == cc)
            if cc_mask.sum() == 0: continue
            AGB_ChangeClass_total[:,cc-1] += scipy.stats.binned_statistic(tt[cc_mask].flatten(), AGB_change[cc_mask].flatten(), bins=np.arange(0,722), statistic='sum')[0] * (25 * 25 * 0.0001)
            AGB_ChangeClass_area[:,cc-1] += scipy.stats.binned_statistic(tt[cc_mask].flatten(), np.ones_like(AGB_change[cc_mask].flatten()), bins=np.arange(0,722), statistic='sum')[0] * (25 * 25 * 0.0001)
    
    df = pd.DataFrame({'AGB_t1_total':AGB_t1_total, 'AGB_t2_total':AGB_t2_total,'TravelTime':np.arange(721),'Area':AGB_area})
    
    df['AGB_t1_mean'] = df['AGB_t1_total'] / df['Area']
    df['AGB_t2_mean'] = df['AGB_t2_total'] / df['Area']
    df['AGB_change_mean'] = (df['AGB_t2_total'] - df['AGB_t1_total']) / df['Area']
    
    for cc in range(1,7):
        df['CC_total_%s'%str(cc)] = AGB_ChangeClass_total[:,cc-1]
        df['CC_area_%s'%str(cc)] = AGB_ChangeClass_area[:,cc-1]
        df['CC_mean_%s'%str(cc)] = df['CC_total_%s'%str(cc)] / df['CC_area_%s'%str(cc)]
        df['CC_proportion_%s'%str(cc)] = df['CC_total_%s'%str(cc)] / AGB_ChangeClass_total.sum(axis=1)
    
    return df



# Options

city_name = 'DarEsSalaam'
input_dir = '/home/sbowers3/DATA/acacia/outputs/%s'%city_name

# H1
df = getTotalAGBChangeFromGeotiff(input_dir)

plt.plot(df['TravelTime'][10:],df['AGB_t1_mean'][10:], label= '2007')
plt.plot(df['TravelTime'][10:],df['AGB_t2_mean'][10:], label= '2016')
plt.xlabel('Travel time (mins)')
plt.ylabel('Mean AGB availability (tC/ha)')
plt.legend()
plt.show()


plt.plot(df['TravelTime'][10:],df['AGB_t1_total'][10:], label= '2007')
plt.plot(df['TravelTime'][10:],df['AGB_t2_total'][10:], label= '2016')
plt.xlabel('Travel time (mins)')
plt.ylabel('Total AGB availability (tC/ha)')
plt.legend()
plt.show()



[plt.plot(df['TravelTime'][10:], df['CC_proportion_%s'%str(i)][10:],label=str(i)) for i in range(1,7)]
plt.legend()
plt.show()
