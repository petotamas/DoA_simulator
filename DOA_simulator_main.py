# Project : PyArgus
# Author  : Tamas Peto
# License : GNU GPL V3 
# -*- coding: utf-8 -*-

import sys
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

import logging
class MainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__ (self,parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
                
        
        #---> DOA display <---
        
        self.figure_DOA, self.axes_DOA = plt.subplots(1, 1, facecolor='white')                
        #self.figure_DOA.suptitle("DOA estimation", fontsize=16)                
        self.canvas_DOA = FigureCanvas(self.figure_DOA)        
        self.gridLayout_main.addWidget(self.canvas_DOA, 1, 1, 1, 1)              
        
        self.axes_DOA.set_xlabel('Amplitude')
        self.axes_DOA.set_ylabel('Incident angle')
        
        # Connect checkbox signals
        
        # Connect spinbox signals
        self.doubleSpinBox_simulation_update_time.valueChanged.connect(self.set_update_time)
        
        self.spinBox_noa.valueChanged.connect(self.antenna_number_changed)
        self.antenna_number_changed()
        # Processing parameters
       
        self.thetas =  np.linspace(0,360,361)

        self.timer = QTimer()
        self.timer.timeout.connect(self.DOA_demo)
        self.timer.start(self.doubleSpinBox_simulation_update_time.value()*1000)
        

    #-----------------------------------------------------------------
    # 
    #-----------------------------------------------------------------
    def set_update_time(self):        
        self.timer.setInterval(self.doubleSpinBox_simulation_update_time.value()*1000)  
    
    def antenna_number_changed(self):
        uca_unamb_radius = 1/(2*np.sqrt(2*(1-np.cos(np.deg2rad(360/self.spinBox_noa.value())))))
        self.label_uca_unamb_radius.setText("{:1.4f}".format(uca_unamb_radius))
              
    def DOA_demo(self):
        self.logger.debug("-> Running simulation <-")
        
        soi_theta = self.horizontalSlider_source_DOA.value()
        
        M = self.spinBox_noa.value() # Number of antenna elements
        N = 2**self.spinBox_sample_size.value() 
        r = self.doubleSpinBox_UCA_r.value()
        d = self.doubleSpinBox_ULA_d.value()
        K = 1 + self.spinBox_multipath_components.value()
        alphas = [1.0]
        thetas = [soi_theta]
        phases = [0]
        try:
            multipath_alphas_str= self.lineEdit_multipath_amplitudes.text().split(',')
            if self.checkBox_multipath_random_angles.isChecked():
                multipath_angles_str = ""
                for k in range(K-1):
                    theta_rnd=np.random.uniform(0,360)
                    multipath_angles_str += "{:3.1f},".format(theta_rnd)

            if self.checkBox_multipath_random_phases.isChecked():                    
                multipath_phases_str = ""
                for k in range(K-1):
                    phase_rnd=np.random.uniform(0,360)
                    multipath_phases_str += "{:3.1f},".format(phase_rnd)

                self.lineEdit_multipath_phases.setText(multipath_phases_str[:len(multipath_phases_str)-1])
            
            multipath_angles_str= self.lineEdit_multipath_angles.text().split(',')
            multipath_phases_str= self.lineEdit_multipath_phases.text().split(',')
            
            # Add multipath parameters
            for k in range(K-1):
                alphas.append(float(multipath_alphas_str[k]))
                thetas.append(float(multipath_angles_str[k]))
                phases.append(float(multipath_phases_str[k]))
                logging.debug("k: {:d}, alpha:{:f} phi:{:f} theta:{:f}".format(k,alphas[k+1],phases[k+1], thetas[k+1]))
            self.label_status.setText("<span style=\" font-size:8pt; font-weight:600; color:#01df01;\" >Running simulation</span>")
        except:
            K=1
            alphas = [1.0]
            phases = [0]
            thetas = [soi_theta]            
            self.label_status.setText("<span style=\" font-size:8pt; font-weight:600; color:#ff0000;\" >Improper multipath parameters</span>")

        alphas = 10**(np.array(alphas)/10) * np.exp(1j*np.deg2rad(np.array(phases)))
        
        noise_pow = 10**(-1*self.spinBox_snr_dB.value()/10)
        
        # Generate the signal of interest        
        soi = np.random.normal(0,1,N) +1j* np.random.normal(0,1,N)
        
        # Generate multichannel uncorrelated noise
        noise = np.random.normal(0, np.sqrt(noise_pow), (M,N) ) +1j* np.random.normal(0, np.sqrt(noise_pow), (M,N) )
          
        """ SNR debug display  
        pn = np.average(np.abs(noise**2))
        ps = np.average(np.abs(soi**2))
        logging.info("SNR: {:.2f}".format(10*np.log10(ps/pn)))
        """
        
        self.axes_DOA.clear()
        legend=[]
            
        if self.checkBox_en_UCA.checkState():
            #---------------- U C A-------------------
            
            A = np.zeros((M, K), dtype=complex)
            
            for k in range(K):
                A[:,k] = np.exp(1j*2*np.pi*r*np.cos(np.radians(thetas[k]-np.arange(0,M,1)*(360)/M))) # UCA
            
            soi_matrix  = (np.outer( soi, np.inner(A, alphas))).T                 
            
            # Create received signal
            rec_signal = soi_matrix + noise
            
            # Calulcate cross-correlation matrix
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
                self.label_Bartlett_UCA_res.setText("{:.1f}".format(np.argmax(Bartlett)))
                self.label_Bartlett_conf.setText("{:.2f}".format(calculate_doa_papr(Bartlett)))
            else:
                self.label_Bartlett_UCA_res.setText("-")
                self.label_Bartlett_conf.setText("-")
            
            if self.checkBox_en_Capon.checkState():
                Capon = de.DOA_Capon(R, scanning_vectors)
                de.DOA_plot(Capon, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("UCA - Capon")
                self.label_Capon_UCA_res.setText("{:.1f}".format(np.argmax(Capon)))
                self.label_Capon_conf.setText("{:.2f}".format(calculate_doa_papr(Capon)))
            else:
                self.label_Capon_UCA_res.setText("-")
                self.label_Capon_conf.setText("-")
    
            if self.checkBox_en_MEM.checkState():
                MEM = de.DOA_MEM(R, scanning_vectors,  column_select = 0)
                de.DOA_plot(MEM, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("MEM")
                self.label_MEM_UCA_res.setText("{:.1f}".format(np.argmax(MEM)))
                self.label_MEM_conf.setText("{:.2f}".format(calculate_doa_papr(MEM)))
            else:
                self.label_MEM_UCA_res.setText("-")
                self.label_MEM_conf.setText("-")
    
            if self.checkBox_en_MUSIC.checkState():
                MUSIC = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
                de.DOA_plot(MUSIC, self.thetas, log_scale_min = -50, axes=self.axes_DOA)
                legend.append("MUSIC")
                self.label_MUSIC_UCA_res.setText("{:.1f}".format(np.argmax(MUSIC)))
                self.label_MUSIC_conf.setText("{:.2f}".format(calculate_doa_papr(MUSIC)))
            else:
                self.label_MUSIC_UCA_res.setText("-")
                self.label_MUSIC_conf.setText("-")

        
        if self.checkBox_en_ULA.checkState():
            #---------------- U L A-------------------            
            # Prepare Array-response matrix
            A = np.zeros((M, K), dtype=complex)
            
            for k in range(K):
                A[:,k] = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(thetas[k])))                
            
            soi_matrix  = (np.outer( soi, np.inner(A, alphas))).T                 
            
            # Create received signal
            rec_signal = soi_matrix + noise
            
            ## R matrix calculation
            R = de.corr_matrix_estimate(rec_signal.T, imp="fast")
            
            if self.checkBox_en_FBavg.isChecked():
                R = de.forward_backward_avg(R)
            
            # Generate array alignment vector            
            array_alignment = np.arange(0, M, 1) * d
            scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, self.thetas)
                        
            # DOA estimation
            alias_highlight = True # Track thaht aliase regions are already shown
            if self.checkBox_en_Bartlett.checkState():
                Bartlett = de.DOA_Bartlett(R, scanning_vectors)    
                de.DOA_plot(Bartlett, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)                
                legend.append("ULA - Bartlett")
                alias_highlight = False
                self.label_Bartlett_ULA_res.setText("{:.1f}".format(np.argmax(Bartlett[0:180])))
                self.label_Bartlett_conf.setText("{:.2f}".format(calculate_doa_papr(Bartlett[0:180])))
            else:
                self.label_Bartlett_ULA_res.setText("-")
                self.label_Bartlett_conf.setText("-")
            
            if self.checkBox_en_Capon.checkState():
                Capon  = de.DOA_Capon(R, scanning_vectors)
                de.DOA_plot(Capon, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA - Capon")
                alias_highlight = False
                self.label_Capon_ULA_res.setText("{:.1f}".format(np.argmax(Capon[0:180])))
                self.label_Capon_conf.setText("{:.2f}".format(calculate_doa_papr(Capon[0:180])))
            else:
                self.label_Capon_ULA_res.setText("-")
                self.label_Capon_conf.setText("-")

    
            if self.checkBox_en_MEM.checkState():
                MEM = de.DOA_MEM(R, scanning_vectors,  column_select = 0)
                de.DOA_plot(MEM, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA - MEM")
                alias_highlight = False
                self.label_MEM_ULA_res.setText("{:.1f}".format(np.argmax(MEM[0:180])))
                self.label_MEM_conf.setText("{:.2f}".format(calculate_doa_papr(MEM[0:180])))
            else:
                self.label_MEM_ULA_res.setText("-")
                self.label_MEM_conf.setText("-")
    
            if self.checkBox_en_MUSIC.checkState():
                MUSIC = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
                de.DOA_plot(MUSIC, self.thetas, log_scale_min = -50, axes=self.axes_DOA, alias_highlight=alias_highlight, d=d)
                legend.append("ULA - MUSIC")
                alias_highlight = False
                self.label_MUSIC_ULA_res.setText("{:.1f}".format(np.argmax(MUSIC[0:180])))
                self.label_MUSIC_conf.setText("{:.2f}".format(calculate_doa_papr(MUSIC[0:180])))
            else:
                self.label_MUSIC_ULA_res.setText("-")
                self.label_MUSIC_conf.setText("-")

            
        self.axes_DOA.legend(legend)        
        self.canvas_DOA.draw()    

def calculate_doa_papr(DOA_data):
    return 10*np.log10(np.max(np.abs(DOA_data))/np.average(np.abs(DOA_data)))    

app = QApplication(sys.argv)
form = MainWindow()
form.show()
app.exec_()