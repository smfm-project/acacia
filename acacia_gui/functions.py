"""
Custard_functions.py
The supporting functions for Acacia.py
"""
################################################################################
################################################################################
# Import external and internal modules

import os
import numpy as np
import pandas as bb
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import shapefile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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

import scipy.ndimage as ndimage
from scipy.ndimage.measurements import label



################################################################################
################################################################################
def figscatter (clusters, cols, df, dir, cluster_num):

    plt.figure(figsize=(12, 7))

    axp =  plt.subplot2grid((2,3),(0,0),colspan=1, rowspan=1)
    axc =  plt.subplot2grid((2,3),(0,1),colspan=1, rowspan=1)
    ax1 =  plt.subplot2grid((2,3),(1,0),colspan=1, rowspan=1)
    axch =  plt.subplot2grid((2,3),(1,1),colspan=1, rowspan=1)
    axrel =  plt.subplot2grid((2,3),(1,2),colspan=1, rowspan=1)

    for n in cluster_num:

        df_n = df.loc[df['clusters'] == n]

        axp.scatter(df_n['perim'], df_n['area'], s = 10, lw = 0, alpha = 0.5, facecolor = plt.cm.jet(df_n['clusters']/max(df['clusters'])))
        axp.set_title('x: perim; y: area')

        axc.scatter(df_n['convexity'], df_n['rect'], s = 10, lw = 0, alpha = 0.5, facecolor = plt.cm.jet(df_n['clusters']/max(df['clusters'])))
        axc.set_title('x: convexity; y: rect')

        ax1.scatter(df_n['AGB_mean'], df_n['AGB_iqr'], s = 10, lw = 0, alpha = 0.5, facecolor = plt.cm.jet(df_n['clusters']/max(df['clusters'])))
        ax1.set_title('x: AGB_mean; y: AGB_iqr')

        axch.scatter(df_n['AGBCh_mean'], df_n['AGBCh_iqr'], s = 10, lw = 0, alpha = 0.5, facecolor = plt.cm.jet(df_n['clusters']/max(df['clusters'])))
        axch.set_title('x: AGBCh_mean; y: AGBCh_iqr')

        axrel.scatter(df_n['AGBR_mean'], df_n['AGBR_iqr'], s = 10, lw = 0, alpha = 0.5, facecolor = plt.cm.jet(df_n['clusters']/max(df['clusters'])))
        axrel.set_title('x: AGBR_mean; y: AGBR_iqr')

    #plt.show()

    plt.savefig(dir+'figscatter.png')





################################################################################
################################################################################
def figscatterval (clusters, cols, df, dv, dir, cluster_num, maxies):
    plt.figure(figsize=(12, 7))

    axp =  plt.subplot2grid((2,3),(0,0),colspan=1, rowspan=1)
    axc =  plt.subplot2grid((2,3),(0,1),colspan=1, rowspan=1)
    ax1 =  plt.subplot2grid((2,3),(1,0),colspan=1, rowspan=1)
    axch =  plt.subplot2grid((2,3),(1,1),colspan=1, rowspan=1)
    axrel =  plt.subplot2grid((2,3),(1,2),colspan=1, rowspan=1)


    cols = ['perim', 'area', 'convexity', 'rect', 'AGB_mean', 'AGBCh_mean', 'AGBR_mean',  'AGB_iqr', 'AGBCh_iqr', 'AGBR_iqr']


    dv = dv.drop(['Active'], axis=1)
    dv = dv.drop(['Centroid'], axis=1)
    dv = dv.drop(['ID'], axis=1)

    for i in range(len(cols)):
        if dv.columns.values[i] == cols[i]:
            dv[cols[i]] = dv[cols[i]]/maxies[i]
        else:
            print('err')


    for n in cluster_num:

        df_n = df.loc[df['clusters'] == n]
        dv_n = dv.loc[dv['clusters'] == n]

        D = [df_n, dv_n]
        Dmark = ['o', '*']
        Dface = [plt.cm.jet(df_n['clusters']/max(df['clusters'])), 'none']
        Dedge = ['none', plt.cm.jet(dv_n['clusters']/max(df['clusters']))]
        Dsize = [10, 75]

        i = 0
        for d in D:

            axp.scatter(d['perim'], d['area'], marker = Dmark[i], s = Dsize[i], alpha = 0.5, edgecolor = Dedge[i], facecolor = Dface[i])
            axp.set_title('x: perim; y: area')

            axc.scatter(d['convexity'], d['rect'], marker = Dmark[i], s = Dsize[i], alpha = 0.5, edgecolor = Dedge[i], facecolor = Dface[i])
            axc.set_title('x: convexity; y: rect')

            ax1.scatter(d['AGB_mean'], d['AGB_iqr'], marker = Dmark[i], s = Dsize[i], alpha = 0.5, edgecolor = Dedge[i], facecolor = Dface[i])
            ax1.set_title('x: AGB_mean; y: AGB_iqr')

            axch.scatter(d['AGBCh_mean'], d['AGBCh_iqr'], marker = Dmark[i], s = Dsize[i], alpha = 0.5, edgecolor = Dedge[i], facecolor = Dface[i])
            axch.set_title('x: AGBCh_mean; y: AGBCh_iqr')

            axrel.scatter(d['AGBR_mean'], d['AGBR_iqr'], marker = Dmark[i], s = Dsize[i], alpha = 0.5, edgecolor = Dedge[i], facecolor = Dface[i])
            axrel.set_title('x: AGBR_mean; y: AGBR_iqr')

            i += 1

    #plt.show()

    plt.savefig(dir+'figscatter_val.png')





################################################################################
################################################################################
def fig1 (clusters, cols, df_classify, dir):

    plt.figure(figsize=(14, 10))

    axp =  plt.subplot2grid((2,3),(0,0),colspan=1, rowspan=1)
    axa =  axp.twinx()
    axc =  plt.subplot2grid((2,3),(0,1),colspan=1, rowspan=1)
    axr =  axc.twinx()
    ax1 =  plt.subplot2grid((2,3),(1,0),colspan=1, rowspan=1)
    axch =  plt.subplot2grid((2,3),(1,1),colspan=1, rowspan=1)
    axrel =  plt.subplot2grid((2,3),(1,2),colspan=1, rowspan=1)
    ax1i =  ax1.twinx()
    axchi =  axch.twinx()
    axreli =  axrel.twinx()


    axes = [axp, axa, axc, axr, ax1, axch, axrel, ax1i, axchi, axreli]
    twin = [False, True, False, True, False, False, False, True, True, True]
    title = ['o Area - Perim *', 'Area - Perim', 'o Convexity - Rectangularity *', 'Convexity - Rectangularity', 'o mean - AGB1 - iqr *', 'o mean - AGB Change - iqr *', 'o mean - Rel AGB Change - iqr *', True, True, True]

    cols = df_classify.columns.values[:-2]


    for a in range(len(cols)):
        if twin[a] is False:
            axes[a].set_title(title[a])

        cluster_range = np.arange(0,max(clusters)+0, 1)
        c_means = []

        for c in cluster_range:
            c_df = df_classify.loc[df_classify['clusters'] == c+1]
            c_arr = np.asarray(c_df[cols[a]])
            c_mean = np.mean(c_arr); c_means.append(c_mean)
            c_std = np.std(c_arr)
            if twin[a] is False:
                axes[a].scatter(c,c_mean, s = 80, marker = 'o', lw = 0)
            else:
                axes[a].scatter(c,c_mean, s = 80, marker = '*', edgecolor = 'k', lw = 0.4)
            axes[a].plot([c,c],[c_mean-c_std, c_mean+c_std], '-k', lw = 0.2)

        # plot limits
        if min(c_means) > 0:
            axes[a].set_ylim(bottom = 0, top = 1.1*max(c_means))
        elif max(c_means) < 0:
            axes[a].set_ylim(top = 1.1*min(c_means), bottom = 0)


    plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=1.2)
    plt.savefig(dir + 'Figure1.png')


################################################################################
################################################################################
def fig2 (clusters, cluster_as, cols, df_classify, df_valid, dir, maxies):
    plt.figure(figsize=(14, 10))

    axp =  plt.subplot2grid((2,3),(0,0),colspan=1, rowspan=1)
    axa =  axp.twinx()
    axc =  plt.subplot2grid((2,3),(0,1),colspan=1, rowspan=1)
    axr =  axc.twinx()
    ax1 =  plt.subplot2grid((2,3),(1,0),colspan=1, rowspan=1)
    axch =  plt.subplot2grid((2,3),(1,1),colspan=1, rowspan=1)
    axrel =  plt.subplot2grid((2,3),(1,2),colspan=1, rowspan=1)
    ax1i =  ax1.twinx()
    axchi =  axch.twinx()
    axreli =  axrel.twinx()

    axes = [axp, axa, axc, axr, ax1, axch, axrel, ax1i, axchi, axreli]
    twin = [False, True, False, True, False, False, False, True, True, True]
    title = ['o Area - Perim *', 'Area - Perim', 'o Convexity - Rectangularity *', 'Convexity - Rectangularity', 'o mean - AGB1 - iqr *', 'o mean - AGB Change - iqr *', 'o mean - Rel AGB Change - iqr *', True, True, True]

    print (cols)
    cols = df_classify.columns.values[:-2]


    for a in range(len(cols)):

        if twin[a] is False:
            axes[a].set_title(title[a])

        cluster_range = np.arange(0,max(clusters)+0, 1)
        c_means = []

        # now plot the field data
        cc_arr = np.asarray(df_valid[cols[a]])
        for i in range(len(cc_arr)):
            if twin[a] is False:
                axes[a].scatter(cluster_as[i]-1,cc_arr[i]/maxies[a], facecolor = 'k', s = 80, marker = 'o', lw = 0,  alpha = 0.2)
            else:
                axes[a].scatter(cluster_as[i]-1,cc_arr[i]/maxies[a], facecolor = 'k', s = 80, marker = '*', edgecolor = 'k', lw = 0., alpha = 0.2)

        for c in cluster_range:
            c_df = df_classify.loc[df_classify['clusters'] == c+1]
            c_arr = np.asarray(c_df[cols[a]])
            c_mean = np.mean(c_arr); c_means.append(c_mean)
            c_std = np.std(c_arr)
            if twin[a] is False:
                axes[a].scatter(c,c_mean, s = 80, marker = 'o', lw = 0)
            else:
                axes[a].scatter(c,c_mean, s = 80, marker = '*', edgecolor = 'k', lw = 0.4)
            axes[a].plot([c,c],[c_mean-c_std, c_mean+c_std], '-k', lw = 0.2)


        # plot limits
        if min(c_means) > 0:
            axes[a].set_ylim(bottom = 0, top = 1.1*max(c_means))
        elif max(c_means) < 0:
            axes[a].set_ylim(top = 1.1*min(c_means), bottom = 0)


    print ('saved the figure')
    plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=1.2)
    plt.savefig(dir+'Figure2.png')





################################################################################
################################################################################
def subsample(df, clusters, N):
    # for each cluster
    # divide all dimensions each in N sections of length 1/N. N is the resolution!
    # identify the points in each slice of x-dimensional space
    # Keep 1 point in each slice and locate it in the dfby it's identifier
    # the selected column at that index becomes the cluster number

    def collect_folders(start, depth=-1):
        #negative depths means unlimited recursion
        folder_ids = []

        # recursive function that collects all the ids in `acc`
        def recurse(current, depth):
            folder_ids.append(current.id)
            if depth != 0:
                for folder in getChildFolders(current.id):
                    # recursive call for each subfolder
                    recurse(folder, depth-1)

        recurse(start, depth) # starts the recursion
        return folder_ids








################################################################################
################################################################################
def save_df_to_shp(df_classify, data_dir):

    df_fields = df_classify.columns.values

    for f in df_fields[:-1]:
        ChangeID = list(df_classify['ChangeID'])
        values = list(df_classify[f])

        # Open a Shapefile, and get field names
        source = ogr.Open(data_dir+'change_polygons.shp', update=True)
        layer = source.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

        field_defn = ogr.FieldDefn( f, ogr.OFTReal )
        layer.CreateField(field_defn)

        # Populate the fields
        for i in layer: # this is the feature index in the shapefile
            feature_value = i.GetField('ChangeID')
            if ChangeID[0] == feature_value:# if the two values match
                i.SetField( f, values[0]) # udpate the stuff
                layer.SetFeature(i)
                if len(ChangeID) > 1:
                    ChangeID = ChangeID[1:]
                    values = values[1:]
        # Close the Shapefile
        source = None

    # Open a Shapefile, and get field names
    source = ogr.Open(data_dir+'change_polygons.shp', update=True)
    layer = source.GetLayer()
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

    # Populate the fields
    for i in range(len(layer)): # this is the feature index in the shapefile
        feature_value = layer[i].GetField('clusters')
        if feature_value == None:
            layer.DeleteFeature(i)

    # Close the Shapefile
    source = None



################################################################################
################################################################################
def cluster (df_final):

    # Build classifier
    ChangeID = df_final['ChangeID']
    df_classify = df_final.drop(['ChangeID'], axis=1)
    df_classify = df_classify.drop(['Centroid'], axis=1)
    df_classify = df_classify.dropna()

    # Make non-dimensional
    cols = df_classify.columns.values
    maxies = np.zeros(len(cols))
    for a in range(len(cols)):
        if np.mean(df_classify[cols[a]]) > 0:
            maxies[a] = max(df_classify[cols[a]])
            df_classify[cols[a]] = df_classify[cols[a]] / max(df_classify[cols[a]])
        else:
            maxies[a] = min(df_classify[cols[a]])
            df_classify[cols[a]] = df_classify[cols[a]]  / min(df_classify[cols[a]])

    #df_classify['AGB_mean'] = 1 * df_classify['AGB_mean']

    # Setup the number of cutoff distances you want
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(df_classify)
    X_scale = scaler.transform(df_classify)
    pca = sklearn.decomposition.PCA(n_components=8).fit(X_scale)
    X_scale = pca.transform(X_scale)

    # Do linkage
    Z = scipy.cluster.hierarchy.linkage(X_scale, 'ward')

    Dists = np.arange(200, 10, -2)
    d_list = []; ratio_list = []; ratio_selected = 100

    for d in Dists:
        print (d)
        # Make the clusters
        clusters = scipy.cluster.hierarchy.fcluster(Z, d, criterion='distance')

        # Add the culsters to the df
        df_classify['clusters'] = clusters # Now we have 'labelled' data

        # define the range of cluster numbers
        cluster_range = np.arange(1,max(clusters)+1, 1)

        # define the columns
        cols = df_classify.columns.values[:-1]

        # For each class
        print ('Collecting the Davies-Bouldin data')
        All_centroid = []; All_sigma = []
        for c in range(len(cluster_range)):
            # This is the points of that cluster
            this_class = df_classify.loc[df_classify['clusters'] == cluster_range[c]]

            # define variables for the Davies-Bouldin index
            # 1. the cluster centroid
            centroid = []
            for a in range(len(cols)):
                centroid.append(np.mean(this_class[cols[a]]))
            centroid = np.asarray(centroid)

            # 2. the average distance to centroid in the cluster
            dist2centroid = []
            this_arr = np.asarray(this_class)
            for i in range(len(this_arr)):
                distance = 0
                for j in range(len(centroid)):
                    element = (centroid[j]-this_arr[i,j])**2
                    distance += element
                dist = np.sqrt(distance)
                dist2centroid.append(dist)
            sigma = np.mean(dist2centroid)

            All_centroid.append(centroid)
            All_sigma.append(sigma)

        All_centroid = np.asarray(All_centroid)
        All_sigma = np.asarray(All_sigma)

        # calculate the Davies-Bouldin
        D = 0
        for i in range(len(cluster_range)):
            Ri = [0]
            for j in range(len(cluster_range)):
                if i != j:
                    Si = All_sigma[i]
                    Sj = All_sigma[j]

                    distance = 0
                    for k in range(len(All_centroid[i])):
                        element = (All_centroid[i][k] - All_centroid[j][k])**2
                        distance += element
                    Mij = np.sqrt(distance)

                    Rij = (Si+Sj) / Mij
                    Ri.append(Rij)

            Di = max(Ri); D += Di

        N = max(cluster_range)
        DB = (1/N) * D

        if N >=5 and N<=10: # we are within the selected range of classes
            if DB < 15: #We have an acceptable Davies-Bouldin
                if DB/N <= ratio_selected: #we improve the ratio
                    d_list.append(d)
                    ratio_list.append(DB/N)
        if N > 10:
            break

    d_list = np.asarray(d_list)
    ratio_list = np.asarray(ratio_list)
    mini = np.where(ratio_list == min(ratio_list))

    d_final = d_list[mini][0]

    clusters = scipy.cluster.hierarchy.fcluster(Z, d_final, criterion='distance')
    # Add the culsters to the df
    df_classify['clusters'] = clusters # Now we have 'labelled' data
    df_classify['ChangeID'] = ChangeID # Now we have 'labelled' data

    return df_classify, cols, clusters, maxies

################################################################################
################################################################################
def nd_distance(arr1,arr2):
    arr = (arr2 - arr1)**2
    D = np.sqrt(np.sum(arr))
    return D

################################################################################
################################################################################
def cluster_valid(df_valid, df_classify, clusters, maxies):
    # drop the insignificant values
    df_bin = df_valid.drop(['ID'], axis=1)
    df_bin = df_bin.drop(['Centroid'], axis=1)
    df_bin = df_bin.drop(['Active'], axis=1)

    df_ref = df_classify.drop(['ChangeID'], axis=1)
    df_ref = df_ref.drop(['clusters'], axis=1)


    # stages of analysis:
    # 1. Loop 1: for each cluster, calculate centroid and cluster size.
    cluster_range = np.arange(0,max(clusters)+0, 1)
    cluster_loc = []
    for c in cluster_range:
        c_df = df_classify.loc[df_classify['clusters'] == c+1]
        c_arr = np.asarray(c_df); c_arr = c_arr[:, :-2]
        arr_mean = c_arr.mean(axis=0)
        cluster_loc.append(arr_mean)

    # 2. Loop 2: for each object, assign a cluster based on dist to centroid.
    assign_cluster = []
    arr = np.asarray(df_bin)

    cluster_as = []
    for i in range(len(arr)):
        D = []
        for j in range(len(cluster_loc)):
            dist = nd_distance(arr[i]/maxies,cluster_loc[j])
            D.append(dist)
        cluster_as.append(np.where(D == min(D))[0][0]+1)

    return df_valid, cluster_as


################################################################################
################################################################################
def make_type(array, QTH, ATH, contiguity = 'queen'):

    # make the types
    where_sig = np.where(array >= QTH)
    type_array = 0 + 0.*array
    type_array[where_sig] = 1

    assert contiguity in ['rook', 'queen'], "Contiguity must be either 'rook' or 'queen'. Input recieved was <%s>."%str(contiguity)


    # If masked, we use this flag to save the mask for later.
    masked = np.ma.isMaskedArray(type_array)

    # Set masked areas to non-contiguous value
    if masked:
        mask = np.ma.getmaskarray(type_array)
        type_array = type_array.filled(0)

    # Label contigous areas with a number
    if contiguity == 'rook':
        structure = ndimage.generate_binary_structure(2,1) # 4-way connectivity
    elif contiguity == 'queen':
        structure = ndimage.generate_binary_structure(2,2) # 8-way connectivity

    location_id, n_areas = label(type_array, structure = structure)

    # Get count of each value in array
    label_area = np.bincount(location_id.flatten())[1:]

    # Find those IDs that meet minimum area requirements
    include_id = np.arange(1, n_areas + 1)[label_area >= ATH]

    # Get a binary array of location_id pixels that meet the minimum area requirement
    contiguous_area = np.in1d(location_id, include_id).reshape(array.shape).astype(np.bool)

    # Return an array giving values to each area
    location_id[contiguous_area == False] = 0

    # Re-number location_ids 1 to n, given that some unique value shave now been removed
    location_id_unique, location_id_indices = np.unique(location_id, return_inverse = True)
    location_id = np.arange(0, location_id_unique.shape[0], 1)[location_id_indices].reshape(array.shape)


    # Put mask back in if input was a masked array
    if masked:
        contiguous_area = np.ma.array(contiguous_area, mask = mask)
        location_id = np.ma.array(location_id, mask = mask)

    final_type =  0. + 0.*array
    final_type [location_id > 0] = 1

    return final_type



################################################################################
################################################################################
def make_chtype(array, type_array, charray, QTH, QCH, rel):
    # Figure out whether this is relative or not
    if rel is False:
        denom = 1
    else:
        denom = QTH
    QCH = QCH/denom

    # make the change types
    where_sig = np.where(array >= QTH)
    where_sig_g = np.where(charray >= QCH)
    where_nsig_g = np.where(charray >= QCH/2)
    where_sig_l = np.where(charray <= -QCH)
    where_nsig_l = np.where(charray <= -QCH/2)

    ch_type_array = -9999. + 0.*type_array
    ch_type_array[np.where(charray >= 0)] = 0
    ch_type_array[np.where(charray <= 0)] = 0
    ch_type_array[where_nsig_g] = 2
    ch_type_array[where_nsig_l] = -2
    ch_type_array[where_sig_g] = 3
    ch_type_array[where_sig_l] = -3
    ch_type_array[type_array == 0] = -9999.

    return ch_type_array





################################################################################
################################################################################
def getContiguousAreas(data_dir, data0, data, value, min_pixels = 1, min_forest_pixels = 1, contiguity = 'queen'):
    '''
    Get pixels that come from the same contigous area.

    Args:
        data: A numpy array
        value: Pixel value to include in contiguous_area (e.g. True for forest)
        min_area: What minimum area should be included (number of pixels)
        contuguity: Set to rook (4-way) or queen (8-way) connectivity constraint. Defaults to 'queen'.

    Returns:
        A binary array of pixels that meet the conditions
    '''

    assert contiguity in ['rook', 'queen'], "Contiguity must be either 'rook' or 'queen'. Input recieved was <%s>."%str(contiguity)

    # Extract area that meets condition
    binary_array = (data == value) * 1

    # If masked, we use this flag to save the mask for later.
    masked = np.ma.isMaskedArray(binary_array)

    # Set masked areas to non-contiguous value
    if masked:
        mask = np.ma.getmaskarray(binary_array)
        binary_array = binary_array.filled(0)

    # Label contigous areas with a number
    if contiguity == 'rook':
        structure = ndimage.generate_binary_structure(2,1) # 4-way connectivity
    elif contiguity == 'queen':
        structure = ndimage.generate_binary_structure(2,2) # 8-way connectivity

    location_id, n_areas = label(binary_array, structure = structure)

    # Get count of each value in array
    label_area = np.bincount(location_id.flatten())[1:]

    # Find those IDs that meet minimum area requirements
    include_id = np.arange(1, n_areas + 1)[label_area >= min_pixels]

    # Get a binary array of location_id pixels that meet the minimum area requirement
    contiguous_area = np.in1d(location_id, include_id).reshape(data.shape).astype(np.bool)

    # Return an array giving values to each area
    location_id[contiguous_area == False] = 0

    # Re-number location_ids 1 to n, given that some unique value shave now been removed
    location_id_unique, location_id_indices = np.unique(location_id, return_inverse = True)
    location_id = np.arange(0, location_id_unique.shape[0], 1)[location_id_indices].reshape(data.shape)


    # Put mask back in if input was a masked array
    if masked:
        contiguous_area = np.ma.array(contiguous_area, mask = mask)
        location_id = np.ma.array(location_id, mask = mask)

    return contiguous_area, location_id




################################################################################
################################################################################
def buildShapefile(input_dir, shapefile_out, samples_per_pc = 100):
    '''
    '''

    #ds_ChangeIDs = [gdal.Open(infile, 0) for infile in sorted(glob.glob(input_dir+'/ChangeID_'+str(y1)+'_'+str(y2)+'_*.tif'))]
    ds_ChangeIDs = [gdal.Open(infile, 0) for infile in sorted(glob.glob(input_dir+'/ChangeID.tif'))]

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
        if samples_per_pc is None:
            N_samples = 0 # Number of polygons
        else:
            N_samples = samples_per_pc

        # Build ChangeID mask. Select random sample of change events
        ds_mask = getMask(ds_ChangeID)

        gdal.Polygonize(ds_ChangeID.GetRasterBand(1), ds_mask.GetRasterBand(1), outLayer, outField, [], callback=None )

        sample_vals.extend(getSample(ds_ChangeID, samples_per_pc = N_samples))

    #sample = np.isin(np.array([feature.GetField('ChangeID') for feature in outLayer]), np.array(sample_vals))

    # Sub-sample output field
    sampleField = ogr.FieldDefn('sample', ogr.OFTInteger)
    outLayer.CreateField(sampleField)

    count = 0
    for feature in outLayer:
        count +=1
        outLayer.SetFeature(feature)
        if feature.GetField('ChangeID') in sample_vals:
            feature.SetField('sample', 1)
        else:
            feature.SetField('sample', 0)

        feature.SetField('ChangeID', count)

        outLayer.SetFeature(feature)

    outDatasource = None

    return shapefile_out



################################################################################
################################################################################
def outputGeoTiff(data, filename, geo_t, proj, output_dir = os.getcwd(), dtype = 6, nodata = None):
    """
    Writes a GeoTiff file to disk.

    Args:
        data: A numpy array.
        geo_t: A GDAL geoMatrix (ds.GetGeoTransform()).
        proj: A GDAL projection (ds.GetProjection()).
        filename: Specify an output file name.
        output_dir: Optioanlly specify an output directory. Defaults to working directory.
        dtype: gdal data type (gdal.GDT_*). Defaults to gdal.GDT_Float32.
        nodata: The nodata value for the array
    """


    # Get full output path
    output_path = '%s/%s.tif'%(os.path.abspath(os.path.expanduser(output_dir)), filename.rstrip('.tif'))

    # Save image with georeference info
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, data.shape[1], data.shape[0], 1, dtype, options = ['COMPRESS=LZW'])
    ds.SetGeoTransform(geo_t)
    ds.SetProjection(proj)

    # Set nodata
    if nodata != None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)

    # Write data for masked and unmasked arrays
    if np.ma.isMaskedArray(data):
        ds.GetRasterBand(1).WriteArray(data.filled(nodata))
    else:
        ds.GetRasterBand(1).WriteArray(data)
    ds = None


################################################################################
################################################################################
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


################################################################################
################################################################################
def getMask(ds_ChangeID):
    '''
    '''

    ChangeID = ds_ChangeID.GetRasterBand(1).ReadAsArray()

    ds_mask = gdal.GetDriverByName('MEM').Create('', ds_ChangeID.RasterXSize, ds_ChangeID.RasterYSize, 1, gdal.GDT_Byte)
    ds_mask.SetGeoTransform(ds_ChangeID.GetGeoTransform())
    ds_mask.SetProjection(ds_ChangeID.GetProjection())
    ds_mask.GetRasterBand(1).WriteArray(np.logical_and(ChangeID>0, ChangeID!=999999).astype(np.int8))

    return ds_mask


################################################################################
################################################################################
def getFeaturesFromValidShapefile(changeIDShapefile, FIELD, subsample = False):
    '''
    '''

    g = ogr.Open(changeIDShapefile)
    layer = g.GetLayer(0)

    sf = shapefile.Reader(changeIDShapefile)
    shapes = sf.shapes()


    changeID, centroid, perims, areas, convexity, circularity, solidity, rectangularity, centroid_offset, hole_ratio, Activ = [], [], [], [], [], [], [], [], [], [], []

    for feature_n in range(layer.GetFeatureCount()):

        feature = layer.GetFeature(feature_n)

        parts = shapes[feature_n].parts
        parts.append(len(shapes[feature_n].points))
        parts = sorted(list(set(parts)))

        x = feature[FIELD]
        Activ.append(x)
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

        #if this_area < 0: pdb.set_trace() # Problem!
        #if this_perim < 0: pdb.set_trace() # Problem!

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
        centroid.append( [centroid_x[0], centroid_y[0]] )


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

    df = pd.DataFrame({'Centroid':centroid, 'perim':perims, 'area':areas, 'convexity':convexity, 'rect':rectangularity, 'Active': Activ})

    return df

################################################################################
################################################################################
def getFeaturesFromShapefile(changeIDShapefile, subsample = False):
    '''
    '''

    g = ogr.Open(changeIDShapefile)
    layer = g.GetLayer(0)

    sf = shapefile.Reader(changeIDShapefile)
    shapes = sf.shapes()


    changeID, centroid, perims, areas, convexity, circularity, solidity, rectangularity, centroid_offset, hole_ratio = [], [], [], [], [], [], [], [], [], []

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
        centroid.append( [centroid_x[0], centroid_y[0]] )


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

    #df = pd.DataFrame({'ChangeID':changeID, 'perim':perims, 'area':areas, 'convexity':convexity, 'circularity':circularity, 'solidity':solidity, 'rectangularity':rectangularity, 'centroid_offset':centroid_offset, 'hole_ratio':hole_ratio})

    df = pd.DataFrame({'ChangeID':changeID, 'Centroid':centroid, 'perim':perims, 'area':areas, 'convexity':convexity, 'rect':rectangularity})

    #df = df.drop(['hole_ratio'],axis=1)

    return df


################################################################################
################################################################################
def getFeaturesFromValidMergedGeotiff(df, input_dir):
    '''
    '''
    AGB_t1_file = input_dir+'/data1.tif'
    AGBChange_file = input_dir+'/change.tif'
    ChangeID_file = input_dir+'/ChangeID.tif'
    #ChangeType_file = input_dir+'/ChangeType_'+str(y1)+'_'+str(y2)+'.tif'

    #assert len(AGB_t1_files) == len(AGBChange_files) == len(ChangeID_files) == len(ChangeType_files)==len(topo_files), "Inputs must be of the same length"

    ID = df.index.values

    df['ID'] = ID


    change_events_sample = np.unique(df['ID'])

    dfs_out = []

    print ('Doing %s'%ChangeID_file.split('/')[-1])

    ChangeID = gdal.Open(ChangeID_file, 0).ReadAsArray()

    sel = np.isin(ChangeID, change_events_sample)

    #if sel.sum() == 0: continue

    AGB_t1 = gdal.Open(AGB_t1_file, 0).ReadAsArray()
    AGB_change = gdal.Open(AGBChange_file).ReadAsArray()
    #ChangeType = gdal.Open(ChangeType_file).ReadAsArray()




    df_all = pd.DataFrame({'AGB':AGB_t1[sel],'AGBCh':AGB_change[sel],'AGBR':AGB_change[sel]/AGB_t1[sel],'ID':ChangeID[sel]})



    # Add t2 AGB
    #df_all['AGB_t2'] = df_all['AGB_t1'] + df_all['AGBChange']

    # Calculate proportional change
    #df_all['AGBChange_proportion'] = ((df_all['AGBChange'] * -1) / df_all['AGB_t1']) * 100

    # Build a new dataframe, containing one record per change event
    # Begin with mean values
    df_group = df_all.groupby(['ID']).mean()

    # Calculate inter-quartile range
    df_group = df_group.merge(df_all.groupby(['ID']).quantile(0.75) - df_all.groupby(['ID']).quantile(0.25), left_index=True, right_index=True, suffixes = ['_mean', '_iqr'])

    # Add together the total AGB loss
    df_sum = df_all.groupby(['ID']).sum()
    df_sum.columns = [str(col) + '_sum' for col in df_sum.columns]
    #df_group = df_group.merge(df_sum,left_index=True,right_index=True).drop(['AGBChange_proportion_sum'], axis=1)

    # Add ChangeID to column, and re-number events
    df_group.index.name = 'ID'
    df_group.reset_index(inplace=True)

    dfs_out.append(df_group.copy())

    df = df.merge(pd.concat(dfs_out))

    # Get rid of features not to be used in classification
    #df = df.drop(['Topo_sum'],axis=1)

    return df

################################################################################
################################################################################
def getFeaturesFromMergedGeotiff(df, input1, input2):
    '''
    '''
    AGB_t1_file = input1
    AGBChange_file = input2
    ChangeID_file = input1[:-9]+'ChangeID.tif'
    #ChangeType_file = input_dir+'/ChangeType_'+str(y1)+'_'+str(y2)+'.tif'

    #assert len(AGB_t1_files) == len(AGBChange_files) == len(ChangeID_files) == len(ChangeType_files)==len(topo_files), "Inputs must be of the same length"

    change_events_sample = np.unique(df['ChangeID'])

    dfs_out = []

    print ('Doing %s'%ChangeID_file.split('/')[-1])

    ChangeID = gdal.Open(ChangeID_file, 0).ReadAsArray()

    sel = np.isin(ChangeID, change_events_sample)

    #if sel.sum() == 0: continue

    AGB_t1 = gdal.Open(AGB_t1_file, 0).ReadAsArray()
    AGB_change = gdal.Open(AGBChange_file).ReadAsArray()
    #ChangeType = gdal.Open(ChangeType_file).ReadAsArray()


    df_all = pd.DataFrame({'AGB':AGB_t1[sel],'AGBCh':AGB_change[sel],'AGBR':AGB_change[sel]/AGB_t1[sel],'ChangeID':ChangeID[sel]})

    # Add t2 AGB
    #df_all['AGB_t2'] = df_all['AGB_t1'] + df_all['AGBChange']

    # Calculate proportional change
    #df_all['AGBChange_proportion'] = ((df_all['AGBChange'] * -1) / df_all['AGB_t1']) * 100

    # Build a new dataframe, containing one record per change event
    # Begin with mean values
    df_group = df_all.groupby(['ChangeID']).mean()

    # Calculate inter-quartile range
    df_group = df_group.merge(df_all.groupby(['ChangeID']).quantile(0.75) - df_all.groupby(['ChangeID']).quantile(0.25), left_index=True, right_index=True, suffixes = ['_mean', '_iqr'])

    # Add together the total AGB loss
    df_sum = df_all.groupby(['ChangeID']).sum()
    df_sum.columns = [str(col) + '_sum' for col in df_sum.columns]
    #df_group = df_group.merge(df_sum,left_index=True,right_index=True).drop(['AGBChange_proportion_sum'], axis=1)

    # Add ChangeID to column, and re-number events
    df_group.index.name = 'ChangeID'
    df_group.reset_index(inplace=True)

    dfs_out.append(df_group.copy())

    df = df.merge(pd.concat(dfs_out))

    # Get rid of features not to be used in classification
    #df = df.drop(['Topo_sum'],axis=1)

    return df
