# -*- coding: utf-8 -*-

import sys
import os
import time
import numpy as np
# Import graphical user interface packages
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import *
from PyQt5.QtCore import QTimer

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

# Import packages for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from DOA_simulator_layout import Ui_MainWindow


# Import the pyArgus module
from pyargus import directionEstimation as de

class MainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__ (self,parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
                
        
        #---> DOA display <---
        
        self.figure_DOA, self.axes_DOA = plt.subplots(1, 1, facecolor='white')                
        #self.figure_DOA.suptitle("DOA estimation", fontsize=16)                
        self.canvas_DOA = FigureCanvas(self.figure_DOA)        
        self.gridLayout_main.addWidget(self.canvas_DOA, 1, 1, 1, 1)              
        
        self.axes_DOA.set_xlabel('Amplitude')
        self.axes_DOA.set_ylabel('Incident angle')
        
        # Connect checkbox signals
#        self.checkBox_en_ula.stateChanged.connect(self.set_DOA_params)        
        
        # Connect spinbox signals
        #self.doubleSpinBox_filterbw.valueChanged.connect(self.set_iq_preprocessing_params)
        
        
        self.horizontalSlider_source_DOA.valueChanged.connect(self.set_DOA_params)
        # Processing parameters
        self.antenna_number = 4
        self.sample_size = 2**10
        self.uca_r = 0.5
        self.ula_d = 0.5
        self.snr = 100
        self.en_uca = False
        self.en_ula = False
        self.soi_theta = 90
        self.thetas =  np.linspace(0,360,361)
        self.set_DOA_params()  

        # Set default confiuration for the GUI components
        self.set_default_configuration()
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.DOA_demo)
        self.timer.start(1000)
        

    #-----------------------------------------------------------------
    # 
    #-----------------------------------------------------------------
    def set_default_configuration(self):
        self.spinBox_DOA_value.setEnabled(False)
                    
    def set_DOA_params(self):
        """
            Update DOA processing parameters
            
            Callback function of:
                -            
        """
        self.soi_theta = self.horizontalSlider_source_DOA.value()
        #  Set DOA processing option
        """ 
        if self.checkBox_en_DOA_Bartlett.checkState():
            self.module_signal_processor.en_DOA_Bartlett = True
        else:
            self.module_signal_processor.en_DOA_Bartlett = False
            
        if self.checkBox_en_DOA_Capon.checkState():
            self.module_signal_processor.en_DOA_Capon = True
        else:
            self.module_signal_processor.en_DOA_Capon = False
            
        if self.checkBox_en_DOA_MEM.checkState():
            self.module_signal_processor.en_DOA_MEM = True
        else:
            self.module_signal_processor.en_DOA_MEM = False       

        if self.checkBox_en_DOA_MUSIC.checkState():
            self.module_signal_processor.en_DOA_MUSIC = True
        else:
            self.module_signal_processor.en_DOA_MUSIC = False
        
        if self.checkBox_en_DOA_FB_avg.checkState():
            self.module_signal_processor.en_DOA_FB_avg = True
        else:
            self.module_signal_processor.en_DOA_FB_avg = False
       
        self.module_signal_processor.DOA_inter_elem_space = self.doubleSpinBox_DOA_d.value()
        """         
    def DOA_demo(self):
        print("Running simulation")
        
        M = self.spinBox_noa.value()
        N = 2**self.spinBox_sample_size.value()
        r = self.doubleSpinBox_UCA_r.value()
        d = self.doubleSpinBox_ULA_d.value()
                
        noise_pow = 10**(-1*self.spinBox_snr_dB.value()/10)
        
        # Generate the signal of interest        
        soi = np.random.normal(0,1,N) +1j* np.random.normal(0,1,N)
        # Generate multichannel uncorrelated noise
        noise = np.random.normal(0, np.sqrt(noise_pow), (M,N) ) +1j* np.random.normal(0, np.sqrt(noise_pow), (M,N) )
        
        """ SNR display  
        pn = np.average(np.abs(noise**2))
        ps = np.average(np.abs(soi**2))
        print("SNR:",10*np.log10(ps/pn))
        """
        
        self.axes_DOA.clear()
        legend=[]
            
        if self.checkBox_en_UCA.checkState():
            #---------------- U C A-------------------
            # Spatial signiture vector
            a = np.ones(M, dtype=complex)        
            for i in np.arange(0,M,1):   
                 a[i] = np.exp(1j*2*np.pi*r*np.cos(np.radians(self.soi_theta-i*(360)/M))) # UCA   
                 #print("%d - %.2f"%(i,a[i]))
            soi_matrix  = (np.outer( soi, a)).T                 
            
            # Create received signal
            rec_signal = soi_matrix + noise
            
            ## R matrix calculation
            R = de.corr_matrix_estimate(rec_signal.T, imp="fast")
            
            #R = forward_backward_avg(R)
            
            # Generate array alignment vector            
            array_alignment = np.arange(0, M, 1) * d
            scanning_vectors = de.gen_uca_scanning_vectors(M, r, self.thetas)
            
            
            # DOA estimation
            alias_highlight = False # Track thaht aliase regions are already shown
            if self.checkBox_en_Bartlett.checkState():
                Bartlett = de.DOA_Bartlett(R, scanning_vectors)  
                de.DOA_plot(Bartlett, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("UCA - Bartlett")
            
            if self.checkBox_en_Capon.checkState():
                Capon = de.DOA_Capon(R, scanning_vectors)
                de.DOA_plot(Capon, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("UCA - Capon")
    
            if self.checkBox_en_MEM.checkState():
                MEM = de.DOA_MEM(R, scanning_vectors,  column_select = 0)
                de.DOA_plot(MEM, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("MEM")
    
            if self.checkBox_en_MUSIC.checkState():
                MUSIC = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
                de.DOA_plot(MUSIC, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("MUSIC")
        
        if self.checkBox_en_ULA.checkState():
            #---------------- U L A-------------------
            # Spatial signiture vector
            a = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(self.soi_theta)))    
    
      
            soi_matrix  = (np.outer( soi, a)).T                 
            
            # Create received signal
            rec_signal = soi_matrix + noise
            
            ## R matrix calculation
            R = de.corr_matrix_estimate(rec_signal.T, imp="fast")
            
            #R = forward_backward_avg(R)
            
            # Generate array alignment vector            
            array_alignment = np.arange(0, M, 1) * d
            scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, self.thetas)
                        
            # DOA estimation
            alias_highlight = True # Track thaht aliase regions are already shown
            if self.checkBox_en_Bartlett.checkState():
                Bartlett = de.DOA_Bartlett(R, scanning_vectors)    
                de.DOA_plot(Bartlett, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)                
                legend.append("ULA-Bartlett")
                alias_highlight = False
            
            if self.checkBox_en_Capon.checkState():
                Capon  = de.DOA_Capon(R, scanning_vectors)
                de.DOA_plot(Capon, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA-Capon")
                alias_highlight = False
    
            if self.checkBox_en_MEM.checkState():
                MEM = de.DOA_MEM(R, scanning_vectors,  column_select = 0)
                de.DOA_plot(MEM, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA-MEM")
                alias_highlight = False
    
            if self.checkBox_en_MUSIC.checkState():
                MUSIC = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
                de.DOA_plot(MUSIC, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA-MUSIC")
                alias_highlight = False
            
        self.axes_DOA.legend(legend)        
        self.canvas_DOA.draw()    
        
app = QApplication(sys.argv)
form = MainWindow()
form.show()
app.exec_()