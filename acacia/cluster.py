# Load properties of change events from an ALOS tile, calculate properties, and cluster



import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
import scipy.spatial
import scipy.cluster
import shapely.geometry
import shapefile
import skimage.morphology
import sklearn.cluster
import sklearn.preprocessing


import pdb

lat = -7 #-23
lon = 38 #32

# Make replicable
np.random.seed(42)


def getSample(ds_ChangeID, samples_per_pc = 10):
    '''
    Build ChangeID mask dataset. Select random sample of change events
    '''
    
    ChangeID = ds_ChangeID.ReadAsArray()
    
    # Calculate numnber of samples to take from tile (10 samples per 1% change)
    n_samples = np.logical_and((ChangeID > 0),ChangeID!=999999).sum().astype(np.float32) / (4500*4500) * samples_per_pc*10
    
    # Round up or down probablistically
    if n_samples - int(n_samples) > np.random.random():
        n_samples = int(n_samples) + 1
    else:
        n_samples = int(n_samples)
        
    # Get list of all changes with area of each
    uid, count = np.unique(ChangeID[np.logical_and((ChangeID > 0),ChangeID!=999999)], return_counts=True)

    # Get random sample from uids, weighted by area
    if n_samples > uid.shape[0]: n_samples = uid.shape[0]
    
    try:
        sample = np.random.choice(uid, n_samples, replace=False, p=count.astype(np.float)/np.sum(count))
    except:
        sample = np.array([])
        #pdb.set_trace()
    
    # Sample ChangeID
    ChangeID_subsample = np.isin(ChangeID, sample)
    
    # Write random sample to gdal dataset
    ds_mask = gdal.GetDriverByName('MEM').Create('', ds_ChangeID.RasterXSize, ds_ChangeID.RasterYSize, 1, gdal.GDT_Byte)
    ds_mask.SetGeoTransform(ds_ChangeID.GetGeoTransform())
    ds_mask.SetProjection(ds_ChangeID.GetProjection())
    ds_mask.GetRasterBand(1).WriteArray(ChangeID_subsample.astype(np.int8))
    
    return sample.tolist() #ds_mask

def getMask(ds_ChangeID):
    '''
    '''
    
    ChangeID = ds_ChangeID.GetRasterBand(1).ReadAsArray()
    
    ds_mask = gdal.GetDriverByName('MEM').Create('', ds_ChangeID.RasterXSize, ds_ChangeID.RasterYSize, 1, gdal.GDT_Byte)
    ds_mask.SetGeoTransform(ds_ChangeID.GetGeoTransform())
    ds_mask.SetProjection(ds_ChangeID.GetProjection())
    ds_mask.GetRasterBand(1).WriteArray(np.logical_and(ChangeID>0, ChangeID!=999999).astype(np.int8))
    
    return ds_mask


def buildShapefile(input_dir, shapefile_out, samples_per_pc = 100):
    '''
    '''
    
    ds_ChangeIDs = [gdal.Open(infile, 0) for infile in sorted(glob.glob('%s/ChangeID_2007_2016_*.tif'%input_dir))]
    
    shapefile_out = os.path.abspath(os.path.expanduser(shapefile_out))
    
    if type(ds_ChangeIDs) != list: ds_ChangeIDs = [ds_ChangeIDs]
    
    driver = ogr.GetDriverByName("ESRI Shapefile")

    if os.path.exists(shapefile_out):
        driver.DeleteDataSource(shapefile_out)

    outDatasource = driver.CreateDataSource(shapefile_out)
    srs = osr.SpatialReference()
    srs.ImportFromWkt( ds_ChangeIDs[0].GetProjectionRef() )
    
    # Create output layer
    outLayer = outDatasource.CreateLayer(shapefile_out, srs)
    
    # ChangeID output field
    newField = ogr.FieldDefn('ChangeID', ogr.OFTInteger)
    outLayer.CreateField(newField)
    outField = outLayer.GetLayerDefn().GetFieldIndex("ChangeID")

    sample_vals = []
    
    for ds_ChangeID in ds_ChangeIDs:
        
        print 'Doing %s'%ds_ChangeID.GetFileList()[0].split('/')[-1]
        
        #if samples_per_pc is None:
        #    gdal.Polygonize(ds_ChangeID.GetRasterBand(1), None, outLayer, 0, [], callback=None )
        
        #else:
        
        # Build ChangeID mask. Select random sample of change events
        ds_mask = getMask(ds_ChangeID)
        
        gdal.Polygonize(ds_ChangeID.GetRasterBand(1), ds_mask.GetRasterBand(1), outLayer, outField, [], callback=None )
        
        sample_vals.extend(getSample(ds_ChangeID, samples_per_pc = samples_per_pc))
    
    #sample = np.isin(np.array([feature.GetField('ChangeID') for feature in outLayer]), np.array(sample_vals))
    
    # Sub-sample output field
    sampleField = ogr.FieldDefn('sample', ogr.OFTInteger)
    outLayer.CreateField(sampleField)
           
    for feature in outLayer:
        outLayer.SetFeature(feature)
        if feature.GetField('ChangeID') in sample_vals:
            feature.SetField('sample', 1) 
        else:
            feature.SetField('sample', 0)
        outLayer.SetFeature(feature)
        
    outDatasource = None
    
    return shapefile_out



def getFeaturesFromShapefile(changeIDShapefile, subsample = False):
    '''
    '''
        
    g = ogr.Open(changeIDShapefile)
    layer = g.GetLayer(0)

    sf = shapefile.Reader(changeIDShapefile)
    shapes = sf.shapes()

    changeID, perims, areas, convexity, circularity, solidity, rectangularity, centroid_offset, hole_ratio = [], [], [], [], [], [], [], [], []
    
    for feature_n in range(layer.GetFeatureCount()):
        
        feature = layer.GetFeature(feature_n)
        
        # Skip feature if only getting subsample
        if subsample and feature.GetField('sample') == 0: continue
        
        this_change_id = feature.GetField('ChangeID')
        
        if this_change_id == 0 or this_change_id == 999999: continue
                
        changeID.append(this_change_id)
        
        parts = shapes[feature_n].parts
        parts.append(len(shapes[feature_n].points))
        parts = sorted(list(set(parts)))
        
        #external = shapes[feature_n].points[parts[0]:parts[1]]
        #internal = []
        
        points = []
        max_bounding_area = 0
        
        for part in range(len(parts)-1):
            
            these_points = shapes[feature_n].points[parts[part]:parts[part+1]]
            
            points.append(these_points)
            
            these_points_arr = np.array(these_points)
            bounding_area = (these_points_arr[:,0].max() - these_points_arr[:,0].min()) * (these_points_arr[:,1].max() - these_points_arr[:,1].min())
            
            # Store index of largest shape, which will be the external polygon. The remainder are internal
            if max_bounding_area < bounding_area:
                external_shape = part
                max_bounding_area = bounding_area.copy()
        
        poly = shapely.geometry.Polygon(points.pop(external_shape), points)
            
        this_perim = poly.exterior.length * 111000
        this_area = poly.area * (111000**2)
        
        if this_area < 0: pdb.set_trace() # Problem!
        if this_perim < 0: pdb.set_trace() # Problem!
        
        perims.append(int(round(this_perim)))
        areas.append(int(round(this_area)))
        
        hull = poly.exterior.convex_hull
        convexity.append((hull.length * 111000) / this_perim)    
        
        # Circularity ratio
        this_circularity_ratio = (shapely.geometry.Polygon(poly.exterior).area * (111000**2)) / (math.pi * ((((poly.exterior.length * 111000) / math.pi) / 2.) ** 2))
        
        circularity.append(this_circularity_ratio)
        
        # Calulate distance of centroid from shape centre by rectangle
        centroid_x_rect, centroid_y_rect = poly.minimum_rotated_rectangle.centroid.xy
        centroid_x, centroid_y = poly.centroid.xy
        
        euclidean_dist = scipy.spatial.distance.euclidean((centroid_x_rect[0], centroid_y_rect[0]), (centroid_x[0], centroid_y[0])) * 111000
        
        centroid_offset.append(euclidean_dist / this_area)
        
        # Rectangularity
        rectangularity.append(poly.area / poly.minimum_rotated_rectangle.area)
        
        # Solidity
        solidity.append(poly.area / hull.area)
        
        # Hole area ratio
        exterior_area = shapely.geometry.Polygon(poly.exterior).area
        interior_area = 0
        for interior in poly.interiors:
            interior_area = shapely.geometry.Polygon(interior).area
        hole_ratio.append(interior_area / exterior_area)
        
    # Close files
    g = None
    sf = None
    
    df = pd.DataFrame({'ChangeID':changeID, 'perim':perims, 'area':areas, 'convexity':convexity, 'circularity':circularity, 'solidity':solidity, 'rectangularity':rectangularity, 'centroid_offset':centroid_offset, 'hole_ratio':hole_ratio})
    
    df = df.drop(['hole_ratio'],axis=1)
    
    return df


def getFeaturesFromGeotiff(df, input_dir):
    '''
    '''
    
    AGB_t1_files = sorted(glob.glob('%s/AGB_2007_*.tif'%input_dir))
    AGBChange_files = sorted(glob.glob('%s/AGBChange_2007_2016_*.tif'%input_dir))
    ChangeType_files = sorted(glob.glob('%s/ChangeType_2007_2016_*.tif'%input_dir))
    ChangeID_files = sorted(glob.glob('%s/ChangeID_2007_2016_*.tif'%input_dir))
    
    assert len(AGB_t1_files) == len(AGBChange_files) == len(ChangeID_files) == len(ChangeType_files), "Inputs must be of the same length"
    
    change_events_sample = np.unique(df['ChangeID'])
    
    dfs_out = []
    
    for AGB_t1_file, AGBChange_file, ChangeType_file, ChangeID_file in zip(AGB_t1_files, AGBChange_files, ChangeType_files, ChangeID_files):
        
        print 'Doing %s'%ChangeID_file.split('/')[-1]
        
        ChangeID = gdal.Open(ChangeID_file, 0).ReadAsArray()
        
        sel = np.isin(ChangeID, change_events_sample)
                
        if sel.sum() == 0: continue
        
        AGB_t1 = gdal.Open(AGB_t1_file, 0).ReadAsArray()
        AGB_change = gdal.Open(AGBChange_file).ReadAsArray()
        ChangeType = gdal.Open(ChangeType_file).ReadAsArray()
        
        df_all = pd.DataFrame({'AGB_t1':AGB_t1[sel],'AGBChange':AGB_change[sel],'ChangeID':ChangeID[sel]})
        
        # Add t2 AGB
        df_all['AGB_t2'] = df_all['AGB_t1'] + df_all['AGBChange']
        
        # Calculate proportional change
        df_all['AGBChange_proportion'] = ((df_all['AGBChange'] * -1) / df_all['AGB_t1']) * 100
        
        # Build a new dataframe, containing one record per change event
        # Begin with mean values
        df_group = df_all.groupby(['ChangeID']).mean()
        
        # Calculate inter-quartile range
        df_group = df_group.merge(df_all.groupby(['ChangeID']).quantile(0.75) - df_all.groupby(['ChangeID']).quantile(0.25), left_index=True, right_index=True, suffixes = ['_mean', '_iqr'])
        
        # Add together the total AGB loss
        df_sum = df_all.groupby(['ChangeID']).sum()
        df_sum.columns = [str(col) + '_sum' for col in df_sum.columns]
        df_group = df_group.merge(df_sum,left_index=True,right_index=True).drop(['AGBChange_proportion_sum'], axis=1)
              
        # Add ChangeID to column, and re-number events
        df_group.index.name = 'ChangeID'
        df_group.reset_index(inplace=True)
        
        dfs_out.append(df_group.copy())
    
    df = df.merge(pd.concat(dfs_out))
    
    # Get rid of features not to be used in classification
    df = df.drop(['AGBChange_sum','AGB_t1_sum','AGB_t2_sum'],axis=1)
    
    return df  



def rasterizeClasses(df, input_dir):
    '''
    '''
    
    ChangeID_files = sorted(glob.glob('%s/ChangeID_2007_2016_*.tif'%input_dir))
    
    for ChangeID_file in ChangeID_files:
        
        print 'Doing %s'%ChangeID_file.split('/')[-1]
        
        ds = gdal.Open(ChangeID_file, 0)
        ChangeID = ds.GetRasterBand(1).ReadAsArray()
        
        ChangeClass = np.zeros_like(ChangeID, dtype = np.int8)
        
        for cluster in np.unique(df['clusters']):
            ChangeClass[np.isin(ChangeID, df['ChangeID'][df['clusters']==cluster])] = cluster
                
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(ChangeID_file.replace('ChangeID', 'ChangeClass'), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])
        ds_out.SetGeoTransform(ds.GetGeoTransform())
        ds_out.SetProjection(ds.GetProjection())
        ds_out.GetRasterBand(1).WriteArray(ChangeClass)
        ds_out = None



# Load arrays
#filename_AGB_2007 = '%s/AGB_2007_S%sE%s.tif'%(input_dir,str(abs(lat)).zfill(2), str(lon).zfill(3))
#filename_AGBChange = '%s/AGBChange_2007_2016_S%sE%s.tif'%(input_dir,str(abs(lat)).zfill(2), str(lon).zfill(3))
#filename_ChangeID = '%s/ChangeID_2007_2016_S%sE%s.tif'%(input_dir,str(abs(lat)).zfill(2), str(lon).zfill(3))

#ds_AGB_2007 = gdal.Open(filename_AGB_2007, 0)
#ds_AGBChange = gdal.Open(filename_AGBChange, 0)
#ds_ChangeID = gdal.Open(filename_ChangeID, 0)

#AGB_2007 = ds_AGB_2007.ReadAsArray()
#AGBChange = ds_AGBChange.ReadAsArray()
#ChangeID = ds_ChangeID.ReadAsArray()


# Options

city_name = 'DarEsSalaam'

city_names = ['Maputo', 'DarEsSalaam', 'Lusaka']

df_out = []
for city_name in city_names:

    input_dir = '/home/sbowers3/DATA/acacia/outputs/%s'%city_name

    # Write shapefile with random sample of ChangeIDs
    changeIDShapefile = buildShapefile(input_dir, '%s/%s_classified_changes.shp'%(input_dir,city_name), samples_per_pc = 600)
    
    # Determine classification features from shapefile
    df = getFeaturesFromShapefile(changeIDShapefile, subsample = True)
    
    # Add classification features from AGB geotiffs
    df_summary = getFeaturesFromGeotiff(df, input_dir)
    
    # Build classifier
    df_classify = df_summary.drop(['ChangeID'], axis=1)
    df_classify = df_classify.dropna()
    
    df_out.append(df_classify.copy())

df_classify = pd.concat(df_out)

# Combine dataframes

# See: https://datascience.stackexchange.com/questions/26183/clustering-data-to-learned-cluster
import scipy.cluster.hierarchy 
import scipy.spatial.distance


#TODO: Settle on a transform that makes sense, and/or apply transforms to data itself

#scaler = sklearn.preprocessing.StandardScaler().fit(df_classify)
scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(df_classify)
#scaler = sklearn.preprocessing.RobustScaler().fit(df_classify)
#scaler = sklearn.preprocessing.PowerTransformer().fit(df_classify)


X_scale = scaler.transform(df_classify)

pca = sklearn.decomposition.PCA(n_components=8).fit(X_scale)
X_scale = pca.transform(X_scale)

# Do linkage
Z = scipy.cluster.hierarchy.linkage(X_scale, 'ward')

# Assess cluster quality
c, coph_dists = scipy.cluster.hierarchy.cophenet(Z, scipy.spatial.distance.pdist(X_scale))
print c

# Calculate full dendrogram, and plot

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
scipy.cluster.hierarchy.dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()


# Set cutoff for n_clusters, and retrieve
max_d = 118#70#103#117
clusters = scipy.cluster.hierarchy.fcluster(Z, max_d, criterion='distance')



def myplot(score,coeff,cluster,labels=None):
    # From: https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    for cluster in np.unique(clusters):
        plt.scatter((xs * scalex)[clusters==cluster],(ys * scaley)[clusters == cluster])
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'k',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'k', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(X_scale[:,0:2],np.transpose(pca.components_[0:2, :]), clusters, labels=list(df_classify.columns))
plt.show()


# Map cluster assignments back to original frame
df_classify['clusters'] = clusters # Now we have 'labelled' data

# Train classifier for classifier (or test a simpler classifier)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(scaler.transform(df_classify.drop(['clusters'],axis=1)), df_classify['clusters'])


# Next, load in all data

for city_name in city_names:
    
    input_dir = '/home/sbowers3/DATA/acacia/outputs/%s'%city_name
    changeIDShapefile = '/home/sbowers3/DATA/acacia/outputs/%s/%s_classified_changes.shp'%(city_name, city_name)
    
    # Determine classification features from shapefile and GeoTiffs
    df_all = getFeaturesFromShapefile(changeIDShapefile, subsample = False)
    df_all = getFeaturesFromGeotiff(df_all, input_dir)

    # Apply classifier to all shapes
    #class_predict = knn.predict(scaler.transform(df_all.drop(['ChangeID'],axis=1)))
    class_predict = knn.predict(scaler.transform(df_all.drop(['ChangeID'],axis=1)))

    df_all['clusters'] = class_predict

    # Output to shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    g = driver.Open(changeIDShapefile, 1)

    output_field = 'cluster'

    layer = g.GetLayer(0)
    layer.CreateField(ogr.FieldDefn(output_field, ogr.OFTInteger))

    for feature in layer:
        
        this_change_id = feature.GetField('ChangeID')
        
        if this_change_id == 0 or this_change_id == 999999: continue
        
        this_class = df_all['clusters'][df_all['ChangeID']==this_change_id]

        layer.SetFeature(feature)
        
        feature.SetField(output_field, int(this_class))
        
        layer.SetFeature(feature)

    g = None
    
    # Raterize shapefile
    rasterizeClasses(df_all, input_dir)


        
##TODO:
# Determine best way to define n_clusters
# Read Ryan et al. 2014



"""
# Extract labels
labels = clustering.labels_

df_classify['labels'] = labels

# Apply to image
ChangeCluster = np.zeros_like(ChangeID) - 1
for label in np.unique(labels):
    ChangeCluster[np.isin(ChangeID,np.array((df[df['labels'] == label]).index))] = label


ChangeCluster_out = np.zeros_like(ChangeCluster, dtype = np.uint8) + 99
ChangeCluster_out[ChangeCluster >=0] = ChangeCluster[ChangeCluster >=0]


# Output as GeoTiff

driver = gdal.GetDriverByName('GTiff')
ds_dest = driver.Create('/home/sbowers3/DATA/acacia/outputs/ChangeCluster.tif', ds_AGB_2007.RasterYSize, ds_AGB_2007.RasterXSize, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])

ds_dest.SetGeoTransform(ds_AGB_2007.GetGeoTransform())
ds_dest.SetProjection(ds_AGB_2007.GetProjection())

ds_dest.GetRasterBand(1).SetNoDataValue(99)
ds_dest.GetRasterBand(1).WriteArray(ChangeCluster_out)
ds_dest.FlushCache()                     # write to disk
ds_dest = None


# Output to shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
g = driver.Open(changeIDShapefile, 1)

output_field = 'Class'

layer = g.GetLayer(0)
layer.CreateField(ogr.FieldDefn(output_field, ogr.OFTInteger))

for feature_n in range(layer.GetFeatureCount()):
    
    feature = layer.GetFeature(feature_n)

    this_change_id = feature.GetField('ChangeID')
    
    if this_change_id == 0 or this_change_id == 999999: continue
    
    this_class = df2['labels'][this_change_id]
    
    feature.SetField(output_field, this_class)
    
    layer.SetFeature(feature)

g = None


"""







"""
#####################################
### Try a PCA/K-means alternative ###
#####################################

# Decompose
import sklearn.decomposition
df_kmeans = df_all.drop(['ChangeID','clusters','centroid_offset','hole_ratio','solidity'],axis=1)
df_kmeans['area'] = np.log(df_kmeans['area'])
df_kmeans['perim'] = np.log(df_kmeans['perim'])

scaler = sklearn.preprocessing.StandardScaler().fit(df_kmeans)
X_scale2 = scaler.transform(df_kmeans)

pca = sklearn.decomposition.PCA(n_components=8).fit(X_scale2)
kmeans = sklearn.cluster.KMeans(n_clusters=6, random_state=0, n_jobs=16).fit(pca.transform(X_scale2))

df_all['clusters'] = kmeans.labels_



h = .02
x_min, x_max = X_scale2[:, 0].min() - 1, X_scale2[:, 0].max() + 1
y_min, y_max = X_scale2[:, 1].min() - 1, X_scale2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*0, xx.ravel()*0, xx.ravel()*0, xx.ravel()*0, xx.ravel()*0, xx.ravel()*0])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_scale2[:, 0], X_scale2[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()



n_components = 6
reduced = pca.transform(X_scale2)
for i in range(0, n_components):
    df_kmeans['PC' + str(i + 1)] = reduced[:, i]


import seaborn as sns
# Do a scree plot
ind = np.arange(0, n_components)
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
ax.set_title('Scree plot')
ax.set_xticks(ind)
ax.set_xticklabels(ind)
ax.set_xlabel('Component Number')
ax.set_ylabel('Explained Variance')
plt.show()



df_kmeans['clusters'] = kmeans.labels_
g = sns.lmplot('PC1','PC2',hue='clusters',data=df_kmeans,fit_reg=False,scatter=True,size=7,)
plt.show()



# Plot a variable factor map for the first two dimensions.
(fig, ax) = plt.subplots(figsize=(12, 12))
for i in range(0, len(pca.components_)):
    ax.arrow(0, 0,  # Start the arrow at the origin
             pca.components_[1, i], pca.components_[2, i],  # 0 and 1 correspond to dimension 1 and 2
             head_width=0.1,head_length=0.1)
    plt.text(pca.components_[1, i] + 0.05, pca.components_[2, i] + 0.05, df_kmeans.columns.values[i])
 
an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map')
plt.show()




"""

