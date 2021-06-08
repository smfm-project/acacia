#####################

#from fbs_runtime.application_context import ApplicationContext
#from PyQt5.QtWidgets import QMainWindow

from __future__ import division
from PyQt5 import QtCore, QtWidgets, QtGui

from acacia_ui import Ui_MyPluginDialogBase  # importing our generated file
import sys
import os
import numpy as np
import platform
import pickle
from osgeo import gdal, ogr, osr


import functions as fn



#####
# A bunch of functions

def error_msg(name_str, info_str):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(name_str)
    msg.setInformativeText(info_str)
    msg.setWindowTitle("Error")
    msg.exec_()


def success_msg():
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText("I'm all done!")
    msg.setInformativeText("Have a nice day :-) ")
    msg.setWindowTitle("Success")
    msg.exec_()


def is_digit(n):
    try:
        int(n)
        return True
    except ValueError:
        return False



def save_params(INP, OUT,LAT, LON, SIZE, bool_F_prop, bool_F_chge, Y1, Y2, bool_Gammao, bool_AGB, bool_WCV, bool_BCH, bool_FCH, bool_RSK, bool_FIL, SUB, POL, ATH, BTH, ACH, BCH, PCH):
    file = open(OUT+"parameters.txt","w")

    file.write("Input_folder = " + INP + " \n")
    file.write("Output_folder = " + OUT + " \n")
    file.write("Latitude = " + LAT + " deg \n")
    file.write("Longitude = " + LON + " deg \n")
    file.write("Tile_size = " + SIZE + " deg \n")
    file.write("Forest_properties = " + str(bool_F_prop) + " \n")
    file.write("Forest_change = " + str(bool_F_chge) + " \n")
    file.write("Year 1 = " + Y1 + " \n")
    file.write("Year 2 = " + Y2 + " \n")
    file.write("Gamma0 = " + str(bool_Gammao) + " \n")
    file.write("AGB = " + str(bool_AGB) + " \n")
    file.write("Forest_cover = " + str(bool_WCV) + " \n")
    file.write("Biomass_change = " + str(bool_BCH) + " \n")
    file.write("Forest_cover_change = " + str(bool_FCH) + " \n")
    file.write("Deforestation_risk = " + str(bool_RSK) + " \n")
    file.write("Filter = " + str(bool_FIL) + " \n")
    file.write("Resample_factor = " + SUB + " \n")
    file.write("Polarisation = " + POL + " \n")
    file.write("Area_threshold = " + ATH + " \n")
    file.write("Biomass_threshold = " + BTH + " \n")
    file.write("Area_change_threshold = " + ACH + " \n")
    file.write("Biomass_change_treshold = " + BCH + " \n")
    file.write("Percentage_change_threshold = " + PCH + " \n")

    file.close()



#####
class mywindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MyPluginDialogBase()
        self.ui.setupUi(self)

        # Set up the text wrap in the intro
        self.ui.Help_label.setWordWrap(True)
        self.ui.label_23.setWordWrap(True)

        # Set up the input file button
        self.ui.lineEdit_INP.clear()
        self.ui.pushButton_INP.clicked.connect(self.get_input_raster)

        # Set up the input file button
        self.ui.lineEdit_val.clear()
        self.ui.pushButton_val.clicked.connect(self.get_val)

        # Set up the ouput folder button
        self.ui.lineEdit_OUT.clear()
        self.ui.pushButton_OUT.clicked.connect(self.get_output_dir)

        # Set up the help button
        self.ui.ManualButton.clicked.connect(self.helplink)

        # Set-up default data location
        self.ui.lineEdit_INP.setText(os.getcwd() + '/data/data1.tif')
        self.ui.lineEdit_INP_2.setText(os.getcwd() + '/data/change.tif')
        self.ui.lineEdit_OUT.setText(os.getcwd() + '/data/')

        #set default values
        self.ui.lineEdit_num.setText('100')
        self.ui.lineEdit_ATH.setText('1')
        self.ui.lineEdit_ACH.setText('1')
        self.ui.lineEdit_ATH.setText('1')
        self.ui.lineEdit_PCH.setText('10')
        self.ui.lineEdit_QCH.setText('5')

        # Set up the logos
        SMFM_label = self.ui.label_21
        SMFM_pixmap = QtGui.QPixmap('gui/Logos/SMFM_Logo.png')
        SMFM_label.setScaledContents(True); SMFM_label.setPixmap(SMFM_pixmap)

        UoE_label = self.ui.label_28
        UoE_pixmap = QtGui.QPixmap('gui/Logos/UoE_Logo.jpg')
        UoE_label.setScaledContents(True); UoE_label.setPixmap(UoE_pixmap)

        LTS_label = self.ui.label_24
        LTS_pixmap = QtGui.QPixmap('gui/Logos/LTS_Logo.png')
        LTS_label.setScaledContents(True); LTS_label.setPixmap(LTS_pixmap)

        white_label = self.ui.label_25
        white_pixmap = QtGui.QPixmap('gui/Logos/white.jpg')
        white_label.setScaledContents(True); white_label.setPixmap(white_pixmap)

        white_label = self.ui.label_26
        white_pixmap = QtGui.QPixmap('gui/Logos/white.jpg')
        white_label.setScaledContents(True); white_label.setPixmap(white_pixmap)


        # Set-up the help buttons
        self.ui.helpButton_num.clicked.connect(self.num_help)
        self.ui.helpButton_val.clicked.connect(self.val_help)
        self.ui.helpButton_field.clicked.connect(self.field_help)
        self.ui.helpButton_ath.clicked.connect(self.ath_help)
        self.ui.helpButton_bth.clicked.connect(self.bth_help)
        self.ui.helpButton_ach.clicked.connect(self.ach_help)
        self.ui.helpButton_BCH.clicked.connect(self.BCH_help)
        self.ui.helpButton_pch.clicked.connect(self.pch_help)
        self.ui.helpButton_REL.clicked.connect(self.rel_help)


    def get_input_raster(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder ")
        dirname = dirname + "/"
        filename  = dirname + "data1.tif"
        filename2  = dirname + "change.tif"
        self.ui.lineEdit_INP.setText(filename)
        self.ui.lineEdit_INP_2.setText(filename2)
        self.ui.lineEdit_OUT.setText(dirname)


    def get_val(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder ")
        dirname = dirname + "/"
        filename  = dirname + "validation.shp"
        self.ui.lineEdit_val.setText(filename)

    def get_output_dir(self):
        filename = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder ")
        filename = filename + "/"
        self.ui.lineEdit_OUT.setText(filename)


    def helplink(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://biota.readthedocs.io/en/latest/'))


    # help functions
    def num_help(self):
        self.ui.Help_label.setText('Radar backscatter from ALOS data. Biomass is calculated from horizontal-send and horizontal-receive (HV) polarisation')
    def val_help(self):
        self.ui.Help_label.setText('Input your validation data polygon shapefile. Make sure all the polygons are in the data extent.')
    def field_help(self):
        self.ui.Help_label.setText('Select the field name for your classification')
    def ath_help(self):
        self.ui.Help_label.setText('Minimum area for the forest cover to consider a patch of pixels to be forested.')
    def bth_help(self):
        self.ui.Help_label.setText('Minimum above-ground biomass density for the forest cover to consider a patch of pixels to be forested.')
    def ach_help(self):
        self.ui.Help_label.setText('Minimum change in forested area for the change algorithm to classify biomass loss as deforestation.')
    def BCH_help(self):
        self.ui.Help_label.setText('Minimum change in forested biomass for the change algorithm to classify biomass loss as deforestation.')
    def pch_help(self):
        self.ui.Help_label.setText('Minimum relative change in forested biomass for the change algorithm to classify biomass loss as deforestation.')
    def rel_help(self):
        self.ui.Help_label.setText('Choose between a dimensional (Quantity change) or relative (Percentage change) approach')


    def reject(self):
        self.close()

    def accept(self):
        # Reset the bars
        self.ui.progressBar_PRC.setValue(0)

        # Get the parameters from the lineEdits and the spinBox
        INP = str(self.ui.lineEdit_INP.text())
        INP2 = str(self.ui.lineEdit_INP_2.text())
        OUT = str(self.ui.lineEdit_OUT.text())
        VAL = str(self.ui.lineEdit_val.text())
        FIELD = str(self.ui.lineEdit_field.text())

        PCH = str(self.ui.lineEdit_PCH.text())
        QTH = str(self.ui.lineEdit_QTH.text())
        QCH = str(self.ui.lineEdit_QCH.text())
        ATH = str(self.ui.lineEdit_ATH.text())
        ACH = str(self.ui.lineEdit_ACH.text())

        # get booleans from checkboxes
        bool_REL = self.ui.checkBox_REL.isChecked()

        bool_area = self.ui.checkBox_area.isChecked()
        bool_perim = self.ui.checkBox_perim.isChecked()
        bool_convexity = self.ui.checkBox_convexity.isChecked()
        bool_rect = self.ui.checkBox_rect.isChecked()
        bool_AGB_mean = self.ui.checkBox_AGB_mean.isChecked()
        bool_AGB_iqr = self.ui.checkBox_AGB_iqr.isChecked()
        bool_AGBCh_mean = self.ui.checkBox_AGBCh_mean.isChecked()
        bool_AGBCh_iqr = self.ui.checkBox_AGBCh_iqr.isChecked()
        bool_AGBR_mean = self.ui.checkBox_AGBR_mean.isChecked()
        bool_AGBR_iqr = self.ui.checkBox_AGBR_iqr.isChecked()


        # Set up the paths
        dir = INP.split('/')[:-1]; dir = '/'.join(dir)+'/'


        # Check the presence/format of all required inputs and print error messages
        good2run = True

        if INP == "" or os.path.isfile(INP) == False :
            error_msg('Input File Error', 'Please enter a valid input file'); good2run = False
        if INP2 == "" or os.path.isfile(INP2) == False :
            error_msg('Input File Error', 'Please enter a valid input file'); good2run = False
        elif OUT == "" or os.path.isdir(OUT) == False:
            error_msg('Output Directory Error', 'Please enter a valid output directory'); good2run = False

        if len(self.ui.lineEdit_num.text())>0:
            NUM = int(self.ui.lineEdit_num.text())
        else:
            error_msg('Number of Elements Warning', 'Not subsampling data might crash ACACIA')
            NUM = 'none'


        # Check the validity of optional inputs and restore default if necessary
        if QTH.isdigit() == False:
            self.ui.lineEdit_QTH.setText('10'); BTH = str(self.ui.lineEdit_QTH.text())
        if QCH.isdigit() == False:
            self.ui.lineEdit_QCH.setText('5'); BCH = str(self.ui.lineEdit_QCH.text())
        if ACH.isdigit() == False:
            self.ui.lineEdit_ACH.setText('2'); ACH = str(self.ui.lineEdit_ACH.text())
        if ATH.isdigit() == False:
            self.ui.lineEdit_ATH.setText('1'); ATH = str(self.ui.lineEdit_ATH.text())
        if PCH.isdigit() == False:
            self.ui.lineEdit_PCH.setText('10'); PCH = str(self.ui.lineEdit_PCH.text())

        # Run the code
        # (for now, make sure you have the biota environment activated)
        print ('Are we good to go?', good2run)

        if good2run == True:

            # Open the rasters
            raster = gdal.Open(INP, gdal.GA_ReadOnly)
            data = raster.GetRasterBand(1); array = data.ReadAsArray()
            chraster= gdal.Open(INP2, gdal.GA_ReadOnly)
            chdata = chraster.GetRasterBand(1); charray = chdata.ReadAsArray()

            # Make the type
            type_array = fn.make_type(array, int(QTH), int(ATH))
            fn.outputGeoTiff(type_array, 'Forest_cover.tif', raster.GetGeoTransform(), raster.GetProjection(), output_dir = dir, dtype = gdal.GDT_Int32, nodata = None)

            # Make the change type
            if bool_REL is False:
                ch_type_array = fn.make_chtype(array, type_array, charray, int(QTH), int(QCH), bool_REL)
            else:
                ch_type_array = fn.make_chtype(array, type_array, charray, int(QTH), int(PCH), bool_REL)

            # Get the contiguous areas
            contiguous_area, location_id = fn.getContiguousAreas(dir, array, ch_type_array, value = -3, min_pixels = int(ACH), min_forest_pixels = int(ATH), contiguity = 'queen')
            self.ui.progressBar_PRC.setValue(10)

            # save the raster
            fn.outputGeoTiff(location_id, 'ChangeID.tif', raster.GetGeoTransform(), raster.GetProjection(), output_dir = dir, dtype = gdal.GDT_Int32, nodata = None)

            #if os.path.isfile(data_dir+'change_polygons.shp') is False:
            changeIDShapefile = fn.buildShapefile(dir, dir+'change_polygons.shp', samples_per_pc = NUM)
            self.ui.progressBar_PRC.setValue(15)

            # Make a panda dataframe to hold the data
            print ('Getting shapes from shp')
            df = fn.getFeaturesFromShapefile(changeIDShapefile, subsample = True)

            print (df)


            quit()

            self.ui.progressBar_PRC.setValue(20)

            # Add classification features from AGB geotiffs
            print ('Getting props from tiff')
            df_final = fn.getFeaturesFromMergedGeotiff(df, INP, INP2)
            self.ui.progressBar_PRC.setValue(30)

            # Drop the columns you don't want
            selcols  = ['perim', 'area', 'convexity', 'rect', 'AGB_mean', 'AGBCh_mean', 'AGBR_mean', 'AGB_iqr', 'AGBCh_iqr', 'AGBR_iqr']
            selbool = [bool_perim, bool_area, bool_convexity, bool_rect, bool_AGB_mean, bool_AGBCh_mean, bool_AGBR_mean, bool_AGB_iqr, bool_AGBCh_iqr, bool_AGBR_iqr]


            for i in range(len(selcols)):
                if selbool[i] is False:
                    df_final = df_final.drop([selcols[i]], axis=1)


            # now cluster stuff
            df_classify, cols, clusters, maxies = fn.cluster(df_final)
            self.ui.progressBar_PRC.setValue(80)


            # Sample points for validation on the field
            df_classify = fn.subsample(df_classify, clusters, N)

            # now save to shp and pickle
            fn.save_df_to_shp(df_classify, OUT)
            with open(OUT+'Classified_df.pkl', 'wb') as handle:
                pickle.dump((df_classify, maxies), handle)

            self.ui.progressBar_PRC.setValue(100)

            print ('doing figures')
            #fn.fig1 (clusters, cols, df_classify, dir)


            if VAL == "" or os.path.isfile(VAL) == False :
                error_msg('Validation File Error', 'Validation file does not exist. We will not process validation')
            else:
                print ('doing validation data')

                # Make a panda dataframe to hold the data
                df_valid = fn.getFeaturesFromValidShapefile(VAL, FIELD, subsample = False)

                # Add classification features from AGB geotiffs
                df_valid = fn.getFeaturesFromValidMergedGeotiff(df_valid, dir)

                # now find the place in the clusters
                df_valid, cluster_as = fn.cluster_valid(df_valid, df_classify, clusters, maxies)
                df_valid['clusters'] = cluster_as

                print (' doing figures')
                #fn.fig2 (clusters, cluster_as, cols, df_classify, df_valid, dir, maxies)
                with open(OUT+'Validation_df.pkl', 'wb') as handle:
                    pickle.dump(df_valid, handle)

            success_msg()

            quit()




            # Print the input parameters in a paramfile
            save_params(INP, OUT,LAT, LON, SIZE, bool_F_prop, bool_F_chge, Y1, Y2, bool_Gammao, bool_AGB, bool_WCV, bool_BCH, bool_FCH, bool_RSK, bool_FIL, SUB, POL, ATH, BTH, ACH, BCH, PCH)



            quit()


# KEEP THIS BIT!
if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    win = mywindow()
    layout = QtWidgets.QVBoxLayout()

    app.setStyle('Fusion')

    win.show()
    app.exec_()
