# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DOA_simulator.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1360, 888)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_main = QtWidgets.QGridLayout()
        self.gridLayout_main.setObjectName("gridLayout_main")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 1)
        self.checkBox_en_ULA = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_en_ULA.setText("")
        self.checkBox_en_ULA.setChecked(True)
        self.checkBox_en_ULA.setObjectName("checkBox_en_ULA")
        self.gridLayout_3.addWidget(self.checkBox_en_ULA, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.checkBox_en_UCA = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_en_UCA.setText("")
        self.checkBox_en_UCA.setObjectName("checkBox_en_UCA")
        self.gridLayout_3.addWidget(self.checkBox_en_UCA, 3, 1, 1, 1)
        self.spinBox_noa = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_noa.setMaximum(200)
        self.spinBox_noa.setProperty("value", 4)
        self.spinBox_noa.setObjectName("spinBox_noa")
        self.gridLayout_3.addWidget(self.spinBox_noa, 0, 1, 1, 1)
        self.doubleSpinBox_ULA_d = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_ULA_d.setMaximum(10.0)
        self.doubleSpinBox_ULA_d.setSingleStep(0.1)
        self.doubleSpinBox_ULA_d.setProperty("value", 0.5)
        self.doubleSpinBox_ULA_d.setObjectName("doubleSpinBox_ULA_d")
        self.gridLayout_3.addWidget(self.doubleSpinBox_ULA_d, 2, 1, 1, 1)
        self.doubleSpinBox_UCA_r = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_UCA_r.setMaximum(10.0)
        self.doubleSpinBox_UCA_r.setSingleStep(0.1)
        self.doubleSpinBox_UCA_r.setProperty("value", 0.5)
        self.doubleSpinBox_UCA_r.setObjectName("doubleSpinBox_UCA_r")
        self.gridLayout_3.addWidget(self.doubleSpinBox_UCA_r, 5, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 5, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 4, 0, 1, 1)
        self.label_uca_unamb_radius = QtWidgets.QLabel(self.groupBox_2)
        self.label_uca_unamb_radius.setObjectName("label_uca_unamb_radius")
        self.gridLayout_3.addWidget(self.label_uca_unamb_radius, 4, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.checkBox_multipath_random_angles = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_multipath_random_angles.setText("")
        self.checkBox_multipath_random_angles.setObjectName("checkBox_multipath_random_angles")
        self.gridLayout_5.addWidget(self.checkBox_multipath_random_angles, 7, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 4, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setObjectName("label_16")
        self.gridLayout_5.addWidget(self.label_16, 7, 0, 1, 1)
        self.spinBox_multipath_components = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_multipath_components.setMaximum(100)
        self.spinBox_multipath_components.setProperty("value", 2)
        self.spinBox_multipath_components.setObjectName("spinBox_multipath_components")
        self.gridLayout_5.addWidget(self.spinBox_multipath_components, 3, 1, 1, 1)
        self.spinBox_sample_size = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_sample_size.setMinimum(3)
        self.spinBox_sample_size.setMaximum(25)
        self.spinBox_sample_size.setProperty("value", 10)
        self.spinBox_sample_size.setObjectName("spinBox_sample_size")
        self.gridLayout_5.addWidget(self.spinBox_sample_size, 1, 1, 1, 1)
        self.lineEdit_multipath_amplitudes = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_multipath_amplitudes.setMaxLength(150)
        self.lineEdit_multipath_amplitudes.setObjectName("lineEdit_multipath_amplitudes")
        self.gridLayout_5.addWidget(self.lineEdit_multipath_amplitudes, 4, 1, 1, 1)
        self.doubleSpinBox_simulation_update_time = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_simulation_update_time.setDecimals(1)
        self.doubleSpinBox_simulation_update_time.setMinimum(0.1)
        self.doubleSpinBox_simulation_update_time.setMaximum(10.0)
        self.doubleSpinBox_simulation_update_time.setProperty("value", 0.1)
        self.doubleSpinBox_simulation_update_time.setObjectName("doubleSpinBox_simulation_update_time")
        self.gridLayout_5.addWidget(self.doubleSpinBox_simulation_update_time, 2, 1, 1, 1)
        self.lineEdit_multipath_angles = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_multipath_angles.setObjectName("lineEdit_multipath_angles")
        self.gridLayout_5.addWidget(self.lineEdit_multipath_angles, 8, 1, 1, 1)
        self.spinBox_snr_dB = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_snr_dB.setMinimum(-100)
        self.spinBox_snr_dB.setMaximum(100)
        self.spinBox_snr_dB.setProperty("value", 10)
        self.spinBox_snr_dB.setObjectName("spinBox_snr_dB")
        self.gridLayout_5.addWidget(self.spinBox_snr_dB, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_5.addWidget(self.label_7, 6, 0, 1, 1)
        self.label_snr = QtWidgets.QLabel(self.groupBox)
        self.label_snr.setObjectName("label_snr")
        self.gridLayout_5.addWidget(self.label_snr, 0, 0, 1, 1)
        self.lineEdit_multipath_phases = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_multipath_phases.setObjectName("lineEdit_multipath_phases")
        self.gridLayout_5.addWidget(self.lineEdit_multipath_phases, 6, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setObjectName("label_11")
        self.gridLayout_5.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        self.label_14.setObjectName("label_14")
        self.gridLayout_5.addWidget(self.label_14, 8, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 3, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 5, 0, 1, 1)
        self.checkBox_multipath_random_phases = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_multipath_random_phases.setText("")
        self.checkBox_multipath_random_phases.setObjectName("checkBox_multipath_random_phases")
        self.gridLayout_5.addWidget(self.checkBox_multipath_random_phases, 5, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setObjectName("label_20")
        self.gridLayout_8.addWidget(self.label_20, 0, 1, 1, 1)
        self.label_Capon_ULA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_Capon_ULA_res.setObjectName("label_Capon_ULA_res")
        self.gridLayout_8.addWidget(self.label_Capon_ULA_res, 2, 1, 1, 1)
        self.checkBox_en_MUSIC = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBox_en_MUSIC.setObjectName("checkBox_en_MUSIC")
        self.gridLayout_8.addWidget(self.checkBox_en_MUSIC, 4, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox_3)
        self.label_19.setObjectName("label_19")
        self.gridLayout_8.addWidget(self.label_19, 0, 0, 1, 1)
        self.label_Capon_UCA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_Capon_UCA_res.setObjectName("label_Capon_UCA_res")
        self.gridLayout_8.addWidget(self.label_Capon_UCA_res, 2, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox_3)
        self.label_21.setObjectName("label_21")
        self.gridLayout_8.addWidget(self.label_21, 0, 2, 1, 1)
        self.label_MUSIC_UCA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_MUSIC_UCA_res.setObjectName("label_MUSIC_UCA_res")
        self.gridLayout_8.addWidget(self.label_MUSIC_UCA_res, 4, 2, 1, 1)
        self.checkBox_en_MEM = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBox_en_MEM.setObjectName("checkBox_en_MEM")
        self.gridLayout_8.addWidget(self.checkBox_en_MEM, 3, 0, 1, 1)
        self.checkBox_en_FBavg = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBox_en_FBavg.setObjectName("checkBox_en_FBavg")
        self.gridLayout_8.addWidget(self.checkBox_en_FBavg, 5, 0, 1, 1)
        self.label_Bartlett_ULA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_Bartlett_ULA_res.setObjectName("label_Bartlett_ULA_res")
        self.gridLayout_8.addWidget(self.label_Bartlett_ULA_res, 1, 1, 1, 1)
        self.label_Bartlett_UCA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_Bartlett_UCA_res.setObjectName("label_Bartlett_UCA_res")
        self.gridLayout_8.addWidget(self.label_Bartlett_UCA_res, 1, 2, 1, 1)
        self.label_MEM_ULA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_MEM_ULA_res.setObjectName("label_MEM_ULA_res")
        self.gridLayout_8.addWidget(self.label_MEM_ULA_res, 3, 1, 1, 1)
        self.checkBox_en_Capon = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBox_en_Capon.setObjectName("checkBox_en_Capon")
        self.gridLayout_8.addWidget(self.checkBox_en_Capon, 2, 0, 1, 1)
        self.label_MUSIC_ULA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_MUSIC_ULA_res.setObjectName("label_MUSIC_ULA_res")
        self.gridLayout_8.addWidget(self.label_MUSIC_ULA_res, 4, 1, 1, 1)
        self.checkBox_en_Bartlett = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBox_en_Bartlett.setEnabled(True)
        self.checkBox_en_Bartlett.setChecked(True)
        self.checkBox_en_Bartlett.setObjectName("checkBox_en_Bartlett")
        self.gridLayout_8.addWidget(self.checkBox_en_Bartlett, 1, 0, 1, 1)
        self.label_MEM_UCA_res = QtWidgets.QLabel(self.groupBox_3)
        self.label_MEM_UCA_res.setObjectName("label_MEM_UCA_res")
        self.gridLayout_8.addWidget(self.label_MEM_UCA_res, 3, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setObjectName("label_9")
        self.gridLayout_8.addWidget(self.label_9, 0, 3, 1, 1)
        self.label_Bartlett_conf = QtWidgets.QLabel(self.groupBox_3)
        self.label_Bartlett_conf.setObjectName("label_Bartlett_conf")
        self.gridLayout_8.addWidget(self.label_Bartlett_conf, 1, 3, 1, 1)
        self.label_Capon_conf = QtWidgets.QLabel(self.groupBox_3)
        self.label_Capon_conf.setObjectName("label_Capon_conf")
        self.gridLayout_8.addWidget(self.label_Capon_conf, 2, 3, 1, 1)
        self.label_MEM_conf = QtWidgets.QLabel(self.groupBox_3)
        self.label_MEM_conf.setObjectName("label_MEM_conf")
        self.gridLayout_8.addWidget(self.label_MEM_conf, 3, 3, 1, 1)
        self.label_MUSIC_conf = QtWidgets.QLabel(self.groupBox_3)
        self.label_MUSIC_conf.setObjectName("label_MUSIC_conf")
        self.gridLayout_8.addWidget(self.label_MUSIC_conf, 4, 3, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.label_status = QtWidgets.QLabel(self.centralwidget)
        self.label_status.setObjectName("label_status")
        self.verticalLayout.addWidget(self.label_status)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_main.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout_main.addWidget(self.label_17, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalSlider_source_DOA = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_source_DOA.setMaximum(360)
        self.horizontalSlider_source_DOA.setProperty("value", 90)
        self.horizontalSlider_source_DOA.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_source_DOA.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_source_DOA.setTickInterval(5)
        self.horizontalSlider_source_DOA.setObjectName("horizontalSlider_source_DOA")
        self.horizontalLayout_2.addWidget(self.horizontalSlider_source_DOA)
        self.spinBox_DOA_value = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_DOA_value.setMaximum(360)
        self.spinBox_DOA_value.setProperty("value", 90)
        self.spinBox_DOA_value.setObjectName("spinBox_DOA_value")
        self.horizontalLayout_2.addWidget(self.spinBox_DOA_value)
        self.gridLayout_main.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_main)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1360, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.horizontalSlider_source_DOA.sliderMoved['int'].connect(self.spinBox_DOA_value.setValue)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Antenna configuration"))
        self.label_2.setText(_translate("MainWindow", "ULA spacing:"))
        self.label_5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Uniform Circular Array antenna arangement</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Enable UCA:"))
        self.label_12.setText(_translate("MainWindow", "Number of antennas:"))
        self.label.setToolTip(_translate("MainWindow", "<html><head/><body><p>Uniform Linar Array antenna arangement</p></body></html>"))
        self.label.setText(_translate("MainWindow", "Enable ULA:"))
        self.label_4.setText(_translate("MainWindow", "UCA radius:"))
        self.label_6.setText(_translate("MainWindow", "Unambiguous radius:"))
        self.label_uca_unamb_radius.setText(_translate("MainWindow", "-"))
        self.groupBox.setTitle(_translate("MainWindow", "Simulation parameters"))
        self.label_3.setText(_translate("MainWindow", "Update time [s]:"))
        self.label_13.setToolTip(_translate("MainWindow", "<html><head/><body><p>Comma separated float values. E.g:&quot;1.0, -25.4, -31.0&quot;<br/>The number of float values specified here should equal to the number of multipath components</p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "Multipath amplitudes [dBc]:"))
        self.label_16.setText(_translate("MainWindow", "Multipath random DoA:"))
        self.lineEdit_multipath_amplitudes.setToolTip(_translate("MainWindow", "<html><head/><body><p>Comma separated float values. E.g:&quot;1.0, -25.4, -31.0&quot;<br/>The number of float values specified here should equal to the number of multipath components</p></body></html>"))
        self.lineEdit_multipath_amplitudes.setText(_translate("MainWindow", "0, -1.5"))
        self.lineEdit_multipath_angles.setToolTip(_translate("MainWindow", "<html><head/><body><p>Comma separated float values. E.g:&quot;-30, 20, 12&quot;<br/>The number of float values specified here should equal to the number of multipath components</p></body></html>"))
        self.lineEdit_multipath_angles.setText(_translate("MainWindow", "-30, 20"))
        self.label_7.setText(_translate("MainWindow", "Multipath phases [deg]:"))
        self.label_snr.setToolTip(_translate("MainWindow", "<html><head/><body><p>Signal to Noise Ratio of the Signal of Interest</p></body></html>"))
        self.label_snr.setText(_translate("MainWindow", "SOI SNR [dB]:"))
        self.lineEdit_multipath_phases.setText(_translate("MainWindow", "123, 218"))
        self.label_11.setText(_translate("MainWindow", "Sample size 2^x:"))
        self.label_14.setToolTip(_translate("MainWindow", "<html><head/><body><p>Comma separated float values. E.g:&quot;-30, 20, 12&quot;<br/>The number of float values specified here should equal to the number of multipath components</p></body></html>"))
        self.label_14.setText(_translate("MainWindow", "Multipath DoA:"))
        self.label_15.setText(_translate("MainWindow", "Multipath components:"))
        self.label_8.setText(_translate("MainWindow", "Multipath random phases:"))
        self.groupBox_3.setTitle(_translate("MainWindow", "DOA estimation processing:"))
        self.label_20.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">ULA res.</span></p></body></html>"))
        self.label_Capon_ULA_res.setText(_translate("MainWindow", "-"))
        self.checkBox_en_MUSIC.setText(_translate("MainWindow", "MUSIC"))
        self.label_19.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Method</span></p></body></html>"))
        self.label_Capon_UCA_res.setText(_translate("MainWindow", "-"))
        self.label_21.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">UCA res.</span></p></body></html>"))
        self.label_MUSIC_UCA_res.setText(_translate("MainWindow", "-"))
        self.checkBox_en_MEM.setText(_translate("MainWindow", "MEM"))
        self.checkBox_en_FBavg.setText(_translate("MainWindow", "Forward-backward avg"))
        self.label_Bartlett_ULA_res.setText(_translate("MainWindow", "-"))
        self.label_Bartlett_UCA_res.setText(_translate("MainWindow", "-"))
        self.label_MEM_ULA_res.setText(_translate("MainWindow", "-"))
        self.checkBox_en_Capon.setText(_translate("MainWindow", "Capon"))
        self.label_MUSIC_ULA_res.setText(_translate("MainWindow", "-"))
        self.checkBox_en_Bartlett.setText(_translate("MainWindow", "Bartlett"))
        self.label_MEM_UCA_res.setText(_translate("MainWindow", "-"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">PAPR</span></p></body></html>"))
        self.label_Bartlett_conf.setText(_translate("MainWindow", "-"))
        self.label_Capon_conf.setText(_translate("MainWindow", "-"))
        self.label_MEM_conf.setText(_translate("MainWindow", "-"))
        self.label_MUSIC_conf.setText(_translate("MainWindow", "-"))
        self.label_status.setText(_translate("MainWindow", "Status:"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Signal of Interest DoA:</span></p></body></html>"))
