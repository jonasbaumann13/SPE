#!python2
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 Jonas Baumann, Steffen Staeck, Christopher Schlesiger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED BY Jonas Baumann, Steffen Staeck, Christopher Schlesiger "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import platform
import numpy as np
import scipy.ndimage
import subprocess as sp # Module to clear the screen of the Terminal
import os
import json
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtCore import pyqtSlot
import traceback
import matplotlib.backends.backend_qt5agg as pltqt
import matplotlib.figure as figure
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
import sys
import time
import sys
sys.path.append('../')
from base.constants import line_E
from base import spe_tools as pt
from base import misc_tools as mt



class PanOnlyToolbar(pltqt.NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in pltqt.NavigationToolbar2QT.toolitems if
                 t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save", )]

#    def __init__(self, *args, **kwargs):
#        super(PanOnlyToolbar, self).__init__(*args, **kwargs)
#        self.layout().takeAt(1)  #or more than 1 if you have more buttons



class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
            
            

        
        
class evares_GUI_main(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        
        self.threadpool = QThreadPool()
        

        if platform.system() == 'Windows':  self.win = True
        else:                               self.win = False
        if not self.win:
            sp.call('clear',shell = True) # clear the screen to get rid of all the print
        self.bg_color = 'light grey'            ### initial color of the GUI
        self.sum_spec = []
        ### initalize main window ###
        super(evares_GUI_main, self).__init__(parent)
        
        ### define os dependent properties ###
        if self.win:    
            win_add = 40
            self.dir_sep = '/'
        else:
            win_add = 30        
            self.dir_sep = '/'
        
        ### define geometry and properties ###   

        plt.rcParams['savefig.dpi'] = 300
        self.screen_properties = QtWidgets.QDesktopWidget().screenGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.window_heigth = 880 + win_add
        self.window_width = 1210
        self.setWindowTitle('evares')
        self.setGeometry(int((self.screen_width-self.window_width)/2),
                         int((self.screen_height-self.window_heigth)/2),
                         self.window_width, self.window_heigth)
        
        self.def_height = 23
        self.def_top = 0 + win_add
        self.def_y0 = 23+40 + win_add
        self.def_y2 = self.def_y0 + 60
        self.def_y1 = self.def_y2 + 440
        
        self.def_x1 = 580
        
        self.def_el = "Cu"
        self.def_fl = "K-L3"
        self.def_button_pressed = 2
        
        self.ecalib_n = 0
        self.ecalib_m = 1
        
        self.spectra_dict = {}
        self.current_spectrum = ''
        
        
        self.rect_inner = None
        self.rect_outer = None
        self.current_roi = ''
        self.current_gexrf_profile = []
        self.new_calc_pm = True
        self.spectra_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.__init_UI()
        self.__init__plot_sumspec()
        self.__init__plot_photonmap()
        self.__init__plot_GEXRF()


        
    def __init_UI(self):  
        ''' define the layout of the UI '''
        ### set actions connected to the menu or toolbar ###
       

        ### set StatusBar ###
        self.statusBar()
                
        ### define labels ###
        
       # self.label_loading_process = QtGui.QLabel('', self)

        self.label_event_no = QtWidgets.QLabel('no. events:',self)
        
        self.label_event_n_minmax = QtWidgets.QLabel('n-px min-max:',self)
        self.label_event_n_minmax_sym = QtWidgets.QLabel('-',self)
        self.label_event_E_minmax = QtWidgets.QLabel('E min-max / ADU:',self)
        self.label_event_E_minmax_sym = QtWidgets.QLabel('-',self)
        self.label_event_x_minmax = QtWidgets.QLabel('x min-max:',self)
        self.label_event_x_minmax_sym = QtWidgets.QLabel('-',self)
        self.label_event_y_minmax = QtWidgets.QLabel('y min-max:',self)
        self.label_event_y_minmax_sym = QtWidgets.QLabel('-',self)
        
        self.label_ecalib_E1 = QtWidgets.QLabel('keV @', self)
        self.label_ecalib_E2 = QtWidgets.QLabel('keV @', self)
        self.label_colorbox = QtWidgets.QLabel('',self)
        self.label_ecalib = QtWidgets.QLabel('energy calibration: mx+n', self)
        self.label_ecalib_m = QtWidgets.QLabel('m =', self)
        self.label_ecalib_n = QtWidgets.QLabel('n =', self)
        
        self.label_gaussian_fit = QtWidgets.QLabel('fit results',self)
        self.label_gaussian_fit.setAlignment(Qt.AlignTop)
        
        self.label_roi_cursor = QtWidgets.QLabel('cursor = 0 ADU', self)
        
        self.label_clim_low = QtWidgets.QLabel('black', self)
        self.label_clim_high = QtWidgets.QLabel('white', self)
        self.label_profile_phi = QtWidgets.QLabel('profile phi / deg', self)
        self.label_roi = QtWidgets.QLabel('ROI:',self)
        self.label_roi_low = QtWidgets.QLabel('(low)', self)
        self.label_roi_high = QtWidgets.QLabel('(high)', self)
        self.label_roi_label = QtWidgets.QLabel('(label)', self)
        self.label_current_roi = QtWidgets.QLabel('(,)', self)

        ### set position of labels ###
        
        n_shift = 20
        E_shift = 0
        xy_shift = 0
        
        self.label_event_no.move(700+xy_shift,self.def_top)
        self.label_event_no.setFixedSize(80,self.def_height)
        
        self.label_event_n_minmax.move(10,self.def_y0-30)
        self.label_event_n_minmax.setFixedSize(100,self.def_height)
        self.label_event_n_minmax_sym.move(96+n_shift,self.def_y0-30)
        self.label_event_n_minmax_sym.setFixedSize(10,self.def_height)
        self.label_event_E_minmax.move(193+E_shift,self.def_y0-30)
        self.label_event_E_minmax.setFixedSize(150,self.def_height)
        self.label_event_x_minmax.move(450+xy_shift,self.def_y0-30)
        self.label_event_x_minmax.setFixedSize(150,self.def_height)
        self.label_event_y_minmax.move(650+xy_shift,self.def_y0-30)
        self.label_event_y_minmax.setFixedSize(150,self.def_height)
        self.label_event_E_minmax_sym.move(345+E_shift,self.def_y0-30)
        self.label_event_E_minmax_sym.setFixedSize(10,self.def_height)
        self.label_event_x_minmax_sym.move(555+xy_shift,self.def_y0-30)
        self.label_event_x_minmax_sym.setFixedSize(10,self.def_height)
        self.label_event_y_minmax_sym.move(755+xy_shift,self.def_y0-30)
        self.label_event_y_minmax_sym.setFixedSize(10,self.def_height)
        self.label_ecalib_E1.move(310,self.def_y1)
        self.label_ecalib_E1.setFixedHeight(self.def_height)
        self.label_ecalib_E2.move(310,self.def_y1+30)
        self.label_ecalib_E2.setFixedHeight(self.def_height)
        self.label_colorbox.move(279,self.def_y0+2)
        self.label_colorbox.setFixedSize(self.def_height-4,self.def_height-4)
        self.label_colorbox.setStyleSheet("QLabel{background-color: black;}")
        self.label_ecalib.move(15,self.def_y1+60)
        self.label_ecalib.setFixedSize(250,self.def_height)    
        self.label_ecalib_m.move(200,self.def_y1+60)
        self.label_ecalib_m.setFixedHeight(self.def_height)
        self.label_ecalib_n.move(305,self.def_y1+60)
        self.label_ecalib_n.setFixedHeight(self.def_height)
        
        self.label_gaussian_fit.move(15,self.def_y1+90)
        self.label_gaussian_fit.resize(250,self.def_height*4+8)
        
        self.label_roi_cursor.move(10,self.def_y2-30)
        self.label_roi_cursor.resize(240,self.def_height)
        
        self.label_clim_low.move(self.def_x1+240,self.def_top)
        if self.win: self.label_clim_low.move(self.def_x1+275,self.def_top)
        self.label_clim_low.setFixedHeight(self.def_height)
        self.label_clim_high.move(self.def_x1+305,self.def_top)
        if self.win: self.label_clim_high.move(self.def_x1+340,self.def_top)
        self.label_clim_high.setFixedHeight(self.def_height)
        self.label_profile_phi.move(self.def_x1+380,self.def_top)
        self.label_profile_phi.resize(110,self.def_height)
        if self.win: 
            self.label_profile_phi.move(self.def_x1+405,self.def_top)
            self.label_profile_phi.resize(100,self.def_height)
        
        self.label_roi.move(self.def_x1+120,self.def_y0)
        self.label_roi.setFixedHeight(self.def_height)
        self.label_roi_low.move(self.def_x1+149,self.def_y0+25)
        self.label_roi_low.setFixedHeight(self.def_height)
        self.label_roi_high.move(self.def_x1+194,self.def_y0+25)
        self.label_roi_high.setFixedHeight(self.def_height)
        self.label_roi_label.move(self.def_x1+239,self.def_y0+25)
        self.label_roi_label.setFixedHeight(self.def_height)    
        self.label_current_roi.move(self.def_x1+15,self.def_y0+25)
        self.label_current_roi.resize(100,self.def_height)
        
        ### define checkboxes ###
        
        self.check_all_spectra = QtWidgets.QCheckBox('for all spectra', self)
        
        ### set position of checkboxes ###
        self.check_all_spectra.move(self.def_x1+280,self.def_y0-25)
        self.check_all_spectra.setFixedSize(120,self.def_height)
        ### define entry fields ###
        
        self.entry_event_no = QtWidgets.QLineEdit('-1',self)
        self.entry_event_n_min = QtWidgets.QLineEdit('1',self)
        self.entry_event_n_max = QtWidgets.QLineEdit('-1',self)
        self.entry_event_E_min = QtWidgets.QLineEdit('0',self)
        self.entry_event_E_max = QtWidgets.QLineEdit('-1',self)
        self.entry_event_x_min = QtWidgets.QLineEdit('0',self)
        self.entry_event_x_max = QtWidgets.QLineEdit('-1',self)
        self.entry_event_y_min = QtWidgets.QLineEdit('0',self)
        self.entry_event_y_max = QtWidgets.QLineEdit('-1',self)
        self.entry_current_file = QtWidgets.QLineEdit('J:/TUB/Projekte/TubeGEXRF/VM/Data/2018/20180621_NiCu/NiCu_9mu_230muA_500ms.spe',self)
        self.entry_ecalib_m = QtWidgets.QLineEdit('1',self)
        self.entry_ecalib_n = QtWidgets.QLineEdit('0',self)
        self.entry_ecalib_ch1 = QtWidgets.QLineEdit('0',self)
        self.entry_ecalib_el1 = QtWidgets.QLineEdit('',self)
        self.entry_ecalib_fl1 = QtWidgets.QLineEdit('',self)
        self.entry_ecalib_ch2 = QtWidgets.QLineEdit('channel',self)
        self.entry_ecalib_el2 = QtWidgets.QLineEdit(self.def_el,self)
        self.entry_ecalib_fl2 = QtWidgets.QLineEdit(self.def_fl,self)
        self.entry_ecalib_E1 = QtWidgets.QLineEdit('0.000', self)
        try:        self.entry_ecalib_E2 = QtWidgets.QLineEdit('{0:.3f}'.format(line_E(self.def_el, self.def_fl)), self)
        except:     self.entry_ecalib_E2 = QtWidgets.QLineEdit('no fp', self)
        
        self.entry_clim_low = QtWidgets.QLineEdit('0',self)
        self.entry_clim_high = QtWidgets.QLineEdit('3',self)
        self.entry_profile_phi = QtWidgets.QLineEdit('0',self)
        self.entry_roi_low = QtWidgets.QLineEdit('',self)
        self.entry_roi_high = QtWidgets.QLineEdit('',self)
        self.entry_roi_label = QtWidgets.QLineEdit('',self)        

        
        ### set position of entry fields ###
        
        self.entry_event_no.move(765+xy_shift,self.def_top)
        self.entry_event_no.setFixedSize(60,self.def_height)
        
        self.entry_event_n_min.move(68+n_shift,self.def_y0-30)
        self.entry_event_n_min.setFixedSize(25,self.def_height)
        self.entry_event_n_max.move(103+n_shift,self.def_y0-30)
        self.entry_event_n_max.setFixedSize(25,self.def_height)
        self.entry_event_E_min.move(286+E_shift,self.def_y0-30)
        self.entry_event_E_min.setFixedSize(50,self.def_height)
        self.entry_event_E_max.move(356+E_shift,self.def_y0-30)
        self.entry_event_E_max.setFixedSize(50,self.def_height)
        self.entry_event_x_min.move(510,self.def_y0-30)
        self.entry_event_x_min.setFixedSize(40,self.def_height)
        self.entry_event_x_max.move(565,self.def_y0-30)
        self.entry_event_x_max.setFixedSize(40,self.def_height)
        self.entry_event_y_min.move(710,self.def_y0-30)
        self.entry_event_y_min.setFixedSize(40,self.def_height)
        self.entry_event_y_max.move(765,self.def_y0-30)
        self.entry_event_y_max.setFixedSize(40,self.def_height)
        self.entry_current_file.move(30,self.def_top)
        self.entry_current_file.resize(270,self.def_height)
        self.entry_ecalib_m.move(230,self.def_y1+60)
        self.entry_ecalib_m.resize(70,self.def_height)
        self.entry_ecalib_n.move(330,self.def_y1+60)
        self.entry_ecalib_n.resize(70,self.def_height)
        self.entry_ecalib_ch1.move(355,self.def_y1)
        self.entry_ecalib_ch1.resize(50,self.def_height)
        self.entry_ecalib_el1.move(115,self.def_y1)
        self.entry_ecalib_el1.resize(30,self.def_height)
        self.entry_ecalib_fl1.move(150,self.def_y1)
        self.entry_ecalib_fl1.resize(45,self.def_height)
        self.entry_ecalib_ch2.move(355,self.def_y1+30)
        self.entry_ecalib_ch2.resize(50,self.def_height)
        self.entry_ecalib_el2.move(115,self.def_y1+30)
        self.entry_ecalib_el2.resize(30,self.def_height)
        self.entry_ecalib_fl2.move(150,self.def_y1+30)
        self.entry_ecalib_fl2.resize(45,self.def_height)
        self.entry_ecalib_E1.move(250,self.def_y1)
        self.entry_ecalib_E1.resize(50,self.def_height)
        self.entry_ecalib_E2.move(250,self.def_y1+30)
        self.entry_ecalib_E2.resize(50,self.def_height)
        
        self.entry_clim_low.move(self.def_x1+280,self.def_top)
        if self.win: self.entry_clim_low.move(self.def_x1+305,self.def_top)
        self.entry_clim_low.resize(20,self.def_height)
        self.entry_clim_high.move(self.def_x1+345,self.def_top)
        if self.win: self.entry_clim_high.move(self.def_x1+370,self.def_top)
        self.entry_clim_high.resize(20,self.def_height)
        self.entry_profile_phi.move(self.def_x1+485,self.def_top)
        self.entry_profile_phi.resize(40,self.def_height)
        self.entry_roi_low.move(self.def_x1+149,self.def_y0)
        self.entry_roi_low.resize(40,self.def_height)
        self.entry_roi_high.move(self.def_x1+194,self.def_y0)
        self.entry_roi_high.resize(40,self.def_height)
        self.entry_roi_label.move(self.def_x1+239,self.def_y0)
        self.entry_roi_label.resize(55,self.def_height)
        
        ### define progress bar ###
        
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.move(480,self.def_top)
        self.progress.resize(200,self.def_height)
        
        ### define buttons ###
        
        self.button_get_filename = QtWidgets.QPushButton(QtGui.QIcon.fromTheme('document-open'),'',self)
        self.button_load = QtWidgets.QPushButton('load', self)
        
        self.button_delete = QtWidgets.QPushButton('del', self)
        self.button_hidespec = QtWidgets.QPushButton('hide', self)
        self.button_plotspec = QtWidgets.QPushButton('plot', self)
        self.button_plotspec_all = QtWidgets.QPushButton('plot all', self)
        self.button_sumspecs = QtWidgets.QPushButton('sum plotted\nspectra', self)
        self.button_save_spec = QtWidgets.QPushButton('save',self)
        
        self.button_get_fpdata_1 = QtWidgets.QPushButton('get',self)
        self.button_get_fpdata_2 = QtWidgets.QPushButton('get',self)
        
        self.button_ecalib = QtWidgets.QPushButton('calibrate', self)
        self.button_calcecalib = QtWidgets.QPushButton('calculate', self)
        
        self.button_fit_gaussian_to_roi = QtWidgets.QPushButton('fit gaussian to ROI', self)
        
        self.button_photonmap = QtWidgets.QPushButton('plot PM', self)
        self.button_photonmap_save = QtWidgets.QPushButton('save PM', self)
        self.button_addroi = QtWidgets.QPushButton('add ROI', self)
        self.button_saverois = QtWidgets.QPushButton('save ROIs', self)
        self.button_loadrois = QtWidgets.QPushButton('load ROIs', self)
        self.button_delroi = QtWidgets.QPushButton('rem. ROI', self)
        
        self.button_save_profile = QtWidgets.QPushButton('save profile', self)
        
        
        ### set functions to buttons ###
        
        self.button_get_filename.clicked.connect(self.get_filename)
        self.button_load.clicked.connect(self.load_data)

        self.button_delete.clicked.connect(self.del_spec)
        self.button_hidespec.clicked.connect(self.hide_spec)
        self.button_plotspec.clicked.connect(self._load_data_and_plot)
        self.button_plotspec_all.clicked.connect(self._load_data_and_plot_all)
        self.button_sumspecs.clicked.connect(self.sum_specs)
        self.button_save_spec.clicked.connect(self.save_spec)
        
        self.button_get_fpdata_1.clicked.connect(self.get_fp_1)
        self.button_get_fpdata_2.clicked.connect(self.get_fp_2)

        self.button_ecalib.clicked.connect(self._update_sum_spec_plot)
        self.button_calcecalib.clicked.connect(self.calculate_calibration)
        
        self.button_fit_gaussian_to_roi.clicked.connect(self._fit_gaussian_to_ROI)
        
        self.button_photonmap.clicked.connect(self.show_photon_map)
        self.button_photonmap_save.clicked.connect(self.save_photon_maps)
        self.button_addroi.clicked.connect(self.add_roi)
        self.button_saverois.clicked.connect(self.save_rois)
        self.button_loadrois.clicked.connect(self.load_rois)
        self.button_delroi.clicked.connect(self.del_roi)
        
        
        self.button_save_profile.clicked.connect(self.save_profile)
        
        ### set position of buttons ###
        
        self.button_get_filename.move(315,self.def_top)
        self.button_get_filename.setFixedSize(30,self.def_height)
        self.button_load.move(350,self.def_top)
        self.button_load.setFixedSize(60,self.def_height)
        
        self.button_delete.move(377,self.def_y0+25)
        self.button_delete.setFixedSize(60,self.def_height)
        self.button_hidespec.move(315,self.def_y0+25)
        self.button_hidespec.setFixedSize(60,self.def_height)
        self.button_plotspec.move(315,self.def_y0)
        self.button_plotspec.setFixedSize(60,self.def_height)
        self.button_plotspec_all.move(377,self.def_y0)
        self.button_plotspec_all.setFixedSize(60,self.def_height)
        self.button_sumspecs.move(440,self.def_y0)
        self.button_sumspecs.setFixedSize(95,self.def_height*2+2)
        self.button_save_spec.move(240,self.def_y0+25)
        self.button_save_spec.setFixedSize(60,self.def_height)
        
        self.button_get_fpdata_1.move(200,self.def_y1)
        self.button_get_fpdata_1.setFixedSize(40,self.def_height)
        self.button_get_fpdata_2.move(200,self.def_y1+30)
        self.button_get_fpdata_2.setFixedSize(40,self.def_height)
        
        self.button_ecalib.move(415,self.def_y1+60)
        self.button_ecalib.setFixedSize(120,self.def_height)
        self.button_calcecalib.move(415,self.def_y1+30)
        self.button_calcecalib.setFixedSize(120,self.def_height)
        
        self.button_fit_gaussian_to_roi.move(385,self.def_y1+90)
        self.button_fit_gaussian_to_roi.setFixedSize(150,self.def_height)
        
        self.button_photonmap.move(self.def_x1+463,self.def_y0)
        self.button_photonmap.setFixedSize(67,self.def_height)
        self.button_photonmap_save.move(self.def_x1+463,self.def_y0+25)
        self.button_photonmap_save.setFixedSize(67,self.def_height)
        self.button_saverois.move(self.def_x1+297+85,self.def_y0)
        self.button_saverois.setFixedSize(80,self.def_height)
        self.button_loadrois.move(self.def_x1+297+85,self.def_y0-25)
        self.button_loadrois.setFixedSize(80,self.def_height)
        self.button_addroi.move(self.def_x1+297,self.def_y0)
        self.button_addroi.setFixedSize(80,self.def_height)
        self.button_delroi.move(self.def_x1+297,self.def_y0+25)
        self.button_delroi.setFixedSize(80,self.def_height)
        self.button_save_profile.move(self.def_x1+380,self.def_y1-14)
        self.button_save_profile.setFixedSize(150,self.def_height)
        
        
        ### define comboBoxes ###
        
        self.comboBox_spectra = QtWidgets.QComboBox(self)
        self.comboBox_rois = QtWidgets.QComboBox(self)
        
        ### add items to comboBoxes ###
        
        ### set functions to comboBoxes ###
        
        self.comboBox_spectra.activated[str].connect(self._select_spectrum)
        self.comboBox_rois.activated[str].connect(self._select_roi)
        
        ### set position of comboBoxes ###
        
        self.comboBox_spectra.move(30,self.def_y0)
        self.comboBox_spectra.setFixedSize(270-25,self.def_height)
        self.comboBox_rois.move(self.def_x1+15,self.def_y0)
        self.comboBox_rois.setFixedSize(90,self.def_height)
        

        
        ### set ToolTips to Widgets ###
        
        #self.button_load.setToolTip('use this for angular .spx files')
        #self.button_exit.setToolTip('Press to exit the program.')

        ### create PSE ###
        
        #self.element_list()
        ### start GUI ###


    def __init__plot_sumspec(self):
        '''define the layout of the plot frame '''
        
        self.figure_matplot_1 = figure.Figure(dpi=80)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        
        self.canvas_matplot_1= pltqt.FigureCanvasQTAgg(self.figure_matplot_1)
        self.canvas_matplot_1.setParent(self)
        self.canvas_matplot_1.move(20,self.def_y2)


        # this is the Navigation widget for matplotlib plots
        self.toolbar_matplot_1 = PanOnlyToolbar(self.canvas_matplot_1, self)
        if self.win:
            self.toolbar_matplot_1.move(0,self.def_y2+390)
        else:
            self.toolbar_matplot_1.move(0,self.def_y2+380)
        
        self.toolbar_matplot_1.setFixedSize(550,40)

        # create an axis
        self.ax_canvas_matplot_1 = self.figure_matplot_1.add_subplot(111)
        self.ax_canvas_matplot_1.set_xlim((0,1000))
        self.ax_canvas_matplot_1.set_ylim((1,1000))
        self._update_sum_spec_plot()


        
        ### establish connections with User action
        self.canvas_matplot_1.mpl_connect('button_press_event',self._on_button_press)
        self.canvas_matplot_1.mpl_connect('button_release_event', self._on_button_release)


    def __init__plot_photonmap(self):
        '''define the layout of the plot frame '''
        
        self.figure_matplot_2 = figure.Figure(dpi=80)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        
        self.canvas_matplot_2= pltqt.FigureCanvasQTAgg(self.figure_matplot_2)
        self.canvas_matplot_2.setParent(self)
        self.canvas_matplot_2.move(self.def_x1+20,self.def_y2)

        # this is the Navigation widget for matplotlib plots
        self.toolbar_matplot_2 = PanOnlyToolbar(self.canvas_matplot_2, self)        
        if self.win:
            self.toolbar_matplot_2.move(self.def_x1,self.def_y2+390)
        else:
            self.toolbar_matplot_2.move(self.def_x1,self.def_y2+380)
        self.toolbar_matplot_2.setFixedSize(550,40)
        
        ### establish connections with User action
        #self.canvas_matplot_2.mpl_connect('button_press_event',self._on_button_press)

        # create an axis
        self.ax_canvas_matplot_2 = self.figure_matplot_2.add_subplot(111)
        self.ax_canvas_matplot_2.set_title('Photon Map')
        self.ax_canvas_matplot_2.set_xlabel('x axis')
        self.ax_canvas_matplot_2.set_ylabel('y axis')
        self.pm = self.ax_canvas_matplot_2.imshow(np.zeros((1024,1024)), interpolation ='nearest', origin='lower', cmap = plt.get_cmap('gray'))
        
        self._update_image_profile_line()

        self.pm.set_clim(0,3)
        self.pm_cb = self.figure_matplot_2.colorbar(self.pm, ax=self.ax_canvas_matplot_2)
        
        ### establish connections with User action
        self.canvas_matplot_2.mpl_connect('button_press_event',self._on_button_press_PM)
        self.canvas_matplot_2.mpl_connect('button_release_event', self._on_button_release_PM)
        

    def __init__plot_GEXRF(self):
        '''define the layout of the plot frame '''
        
        self.figure_matplot_3 = figure.Figure(figsize=(6.4,3), dpi=80)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        
        self.canvas_matplot_3= pltqt.FigureCanvasQTAgg(self.figure_matplot_3)
        self.canvas_matplot_3.setParent(self)
        self.canvas_matplot_3.move(self.def_x1+20,self.def_y1+30-23)
        
        # this is the Navigation widget for matplotlib plots
        self.toolbar_matplot_3 = PanOnlyToolbar(self.canvas_matplot_3, self)        
        if self.win:
            self.toolbar_matplot_3.move(self.def_x1,self.def_y1+275-23)
        else:
            self.toolbar_matplot_3.move(self.def_x1,self.def_y1+265-23)
        self.toolbar_matplot_3.setFixedSize(550,40)

        # create an axis
        self.ax_canvas_matplot_3 = self.figure_matplot_3.add_subplot(111)
        self._update_GEXRF_plot()

        
    def get_filename(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getOpenFileName(self, 'open file', self.current_spectrum, "spe files (*.h5 *.spe *.npy *.npz)")    ### function for a folder-GUI
        try:
            data_path, types = output
        except:
            data_path = output
        if not data_path == '':
            self.entry_current_file.setText(str(data_path))
        
        
    def load_data(self, file_name = []):
        ### get the entered folder-path and load the content into a list folder_content
        try:
            if not file_name:
                data_path = str(self.entry_current_file.text())
                data_path = data_path.replace('\\',os.sep)
                data_path = data_path.replace('/',os.sep)
                if os.path.isdir(data_path): 
                    if data_path[-1] == os.sep:    
                        data_path_h5 = data_path+'*.h5'
                        data_path_npy = data_path+'*.npy'
                        data_path_npz = data_path+'*.npz'
                        data_path_spe = data_path+'*.spe'
                    else:                       
                        data_path_h5 = data_path+os.sep+'*.h5'
                        data_path_npy = data_path+os.sep+'*.npy'
                        data_path_npz = data_path+os.sep+'*.npz'
                        data_path_spe = data_path+os.sep+'*.spe'
                    data_path_list = glob.glob(data_path_h5) + glob.glob(data_path_npy) + glob.glob(data_path_npz) + glob.glob(data_path_spe)
                else:
                    data_path_list = glob.glob(data_path)
            else:
                data_path_list = glob.glob(file_name)
    
            old_indices = []
            for dp in self.spectra_dict:
                old_indices.append(self.spectra_dict[dp]['id'])
                
            # ind = len(self.spectra_dict)
            for i,dp in enumerate(data_path_list):
                # skip entries already loaded
                if not dp in self.spectra_dict.keys():
                    # find free index
                    index = 0
                    while index in old_indices:
                        index += 1
                    old_indices.append(index)
                    
                    # get color for new entry
                    c_ind = index%len(self.spectra_color_list)
                    color = self.spectra_color_list[c_ind]
                    
                    self.spectra_dict[dp]={'id':index,'label_short':[],'sum_spec':[],'n_spec':[],'n1_n2':(),'color':color,'plot':False, 'top':False, 'rois':{}, 'gaussians':{}, 'photon_maps':{}, 'cube':None}
                    
                    # make shure short name does not already exist
                    dp_short = dp.split(os.sep)[-1].split('.')[0]
                    redundant_no = 2
                    dp_tmp = dp_short
                    dp_shorts = [self.spectra_dict[dp]['label_short'] for dp in self.spectra_dict]
                    while dp_tmp in dp_shorts:
                        dp_tmp = dp_short+'('+str(redundant_no)+')'
                        redundant_no+=1
                    dp_short = dp_tmp
                    self.spectra_dict[dp]['label_short']=dp_short
                    
                    # add entry to comboBox and set color
                    self.comboBox_spectra.addItem(dp_short)
                    combo_ind = self.comboBox_spectra.findText(dp_short)
                    self.comboBox_spectra.setItemData(combo_ind,QtGui.QColor(u'#000000'),Qt.TextColorRole)
                    self.current_spectrum = dp
                    
                    # comboBox_items = [self.comboBox_spectra.itemText(i) for i in range(self.comboBox_spectra.count())]
                    # while dp_tmp in comboBox_items:
                    #     dp_tmp = dp_short+'('+str(redundant_no)+')'
                    #     redundant_no+=1
                    # dp_short = dp_tmp
                    # self.spectra_dict[dp]['label_short']=dp_short
                    # self.comboBox_spectra.addItem(dp_short)
                    # self.current_spectrum = dp
                    # self.comboBox_spectra.setItemData(i+ind,QtGui.QColor(u'#000000'),Qt.TextColorRole)
        except:
            print('data not loaded:', sys.exc_info()[0])
            


    def del_spec(self):
        try:
            combo_ind = self.comboBox_spectra.currentIndex()
            self.comboBox_spectra.removeItem(combo_ind)
            self.spectra_dict.pop(self.current_spectrum)
            self._select_spectrum()                      # triggers update_sum_spec_plot
        except:
            print('spectrum not deleted:', sys.exc_info()[0])


    def hide_spec(self):
        try:
            combo_ind = self.comboBox_spectra.currentIndex()
            self.spectra_dict[self.current_spectrum]['plot'] = False
            self.comboBox_spectra.setItemData(combo_ind,QtGui.QColor(u'#000000'),Qt.ForegroundRole)#TextColorRole)
            self._update_sum_spec_plot()
        except:
            print('spectrum not hidden:', sys.exc_info()[0])


        
    def load_spectrum_data(self, file_list, event_no, Emin_Emax, xy_roi, progress_callback):
        print('loading...')
        n1,n2 = self._get_n1_n2()
        results = {}
        for file in file_list:
            if not list(self.spectra_dict[file]['sum_spec']):
                if file.split('.')[-1] == 'spe':
                    sum_spec, n_spec, mode = pt.get_spectrum(file,
                                                                event_no, 
                                                                Emin_Emax=Emin_Emax, 
                                                                xy_roi = xy_roi, 
                                                                progress_bar=progress_callback)
                    results[file] = [sum_spec, n_spec, n1, n2]
                # for hdf5 format load cube if it is not available, yet -- make sure only one cube is loaded -- should be the one with current_spectrum
                elif file.split('.')[-1] in ['h5', 'hdf5', 'npy', 'npz']:
                    if self.spectra_dict[file]['cube'] is None:
                        cube, metadata = pt.load_cube(file)
                    else:
                        cube = self.spectra_dict[file]['cube']

                    sum_spec = pt.get_spectrum_from_cube(cube, e_roi=Emin_Emax, xy_roi=xy_roi)
                    self.spectra_dict[file]['cube'] = None
                    results[file] = [sum_spec, None, n1, n2]
                    if file == self.current_spectrum:
                        self.spectra_dict[file]['cube'] = cube
            else:
                results[file] = [self.spectra_dict[file]['sum_spec'], 
                                      self.spectra_dict[file]['n_spec'],
                                      n1, n2]
        return results
    
        
    def _load_progress(self, percentage):
        self.progress.setValue(percentage)
        
    def _finish_load_data(self, results):
        for file in results:
            [sum_spec, n_spec, n1, n2] = results[file]
            self.spectra_dict[file]['n_spec'] = n_spec
            self.spectra_dict[file]['n1_n2'] = (n1,n2)
            if n2 == -1: n2 = None
            if n_spec is None:  self.spectra_dict[file]['sum_spec'] = sum_spec
            else:               self.spectra_dict[file]['sum_spec'] = np.sum(self.spectra_dict[file]['n_spec'][n1-1:n2],axis=0)
            
            if not 'all' in self.spectra_dict[file]['rois']:
                self.spectra_dict[file]['rois']['all'] = [0,len(sum_spec)]
                self.spectra_dict[file]['photon_maps']['all'] = []
                self.comboBox_rois.addItem("all")
            self.spectra_dict[file]['plot'] = True
            
        self.progress.setValue(1)


    def _plot_spec(self, file_path = None):
        try:
            if file_path is None: file_path = self.current_spectrum
            combo_ind = self.comboBox_spectra.findText(self.spectra_dict[file_path]['label_short'])
            color = self.spectra_dict[file_path]['color']
            self.comboBox_spectra.setItemData(combo_ind,QtGui.QColor(color),Qt.ForegroundRole)
            self._select_spectrum()                              # triggers update_sum_spec_plot                
            self.progress.setValue(0)
        except:
            print('spectrum not plotted:', sys.exc_info()[0])
            
    def _plot_specs(self):
        print("...loading completed!")
        time.sleep(0.1)
        for file_path in self.spectra_dict:
            if self.spectra_dict[file_path]['plot']:
                self.current_spectrum=file_path
                self._plot_spec()
        
    def _load_data_and_plot(self,file_list=None):
        if file_list is None or file_list is False: 
            file_list = [self.current_spectrum]
        event_no = int(float(self.entry_event_no.text()))
        E_min = int(self.entry_event_E_min.text())
        E_max = int(self.entry_event_E_max.text())
        # xy_roi = [[int(self.entry_event_x_min.text()), int(self.entry_event_y_min.text())],
        #           [int(self.entry_event_x_max.text()), int(self.entry_event_y_max.text())]]
        xy_roi = [[int(self.entry_event_y_min.text()), int(self.entry_event_y_max.text())],
                  [int(self.entry_event_x_min.text()), int(self.entry_event_x_max.text())]]
        load_data_worker = Worker(self.load_spectrum_data, file_list, event_no, (E_min, E_max), xy_roi)
        load_data_worker.signals.progress.connect(self._load_progress)
        load_data_worker.signals.result.connect(self._finish_load_data)
        load_data_worker.signals.finished.connect(self._plot_specs)
        self.threadpool.start(load_data_worker)


    def _load_data_and_plot_all(self):
        try:
            self.progress.setValue(1)
            self._load_data_and_plot(self.spectra_dict)
        except:
            print('spectra not plotted:', sys.exc_info()[0])
        
    
    def sum_specs(self):
        try:
            self.progress.setValue(1)
            file_list = []
            for file_path in self.spectra_dict:
                if self.spectra_dict[file_path]['plot'] == True:
                    file_list.append(file_path)
            sum_file = pt.sum_spe_files(file_list,None)#self.progress)
            self.load_data(sum_file)
            self.comboBox_spectra.setCurrentIndex(self.comboBox_spectra.count() - 1)
            self._load_data_and_plot()
        except:
            print('spectra not summed:', sys.exc_info()[0])
    
    def save_spec(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getSaveFileName(self, 'save file', self.current_spectrum)
        try:
            data_path, types = output
        except:
            data_path = output
        if not data_path == '':
            try:
                np.savetxt(data_path,self.spectra_dict[self.current_spectrum]['sum_spec'])
            except:
                print('ERROR: no spectrum calculated, yet',sys.exc_info())    
        
        
    def _select_spectrum(self):
        label = self.comboBox_spectra.currentText()
        for file_path in self.spectra_dict:
            if self.spectra_dict[file_path]['label_short']== str(label):
                self.current_spectrum = file_path
                self.comboBox_spectra.setToolTip(self.current_spectrum)
                if self.spectra_dict[file_path]['n1_n2']:
                    self._set_n1_n2(*self.spectra_dict[file_path]['n1_n2'])
            self.spectra_dict[file_path]['top'] = False
        if len(self.spectra_dict) > 0:              # only, if not all elements are deleted
            self.spectra_dict[self.current_spectrum]['top'] = True
            if self.spectra_dict[self.current_spectrum]['plot'] == True:
                self.label_colorbox.setStyleSheet("QLabel{background-color: "+self.spectra_dict[self.current_spectrum]['color']+";}")                    
            else:
                self.label_colorbox.setStyleSheet("QLabel{background-color: #000000}")
            plt.rcParams["savefig.directory"] = os.path.dirname(self.current_spectrum)
        self._update_roi_list(keep_top=str(self.comboBox_rois.currentText()))             # triggers update_sum_spec_plot
  
    
    def _update_sum_spec_plot(self):
        plotted_spectra = len(self.ax_canvas_matplot_1.lines)
        uncalib = lambda x: (x-self.ecalib_n)/self.ecalib_m
        x_limits=list(map(uncalib,self.ax_canvas_matplot_1.get_xlim()))
        y_limits=self.ax_canvas_matplot_1.get_ylim()
        self.ecalib_m = float(self.entry_ecalib_m.text())
        self.ecalib_n = float(self.entry_ecalib_n.text())
        ecalib = lambda x: self.ecalib_m * x + self.ecalib_n
        
        self.ax_canvas_matplot_1.clear()
        self.ax_canvas_matplot_1.set_xlim(x_limits)
        self.ax_canvas_matplot_1.set_ylim(y_limits)
        
        for file_path in self.spectra_dict:
            if list(self.spectra_dict[file_path]['sum_spec']) and self.spectra_dict[file_path]['plot'] and not self.spectra_dict[file_path]['top']:
                energy_axis = list(map(ecalib,range(len(self.spectra_dict[file_path]['sum_spec']))))
                self.ax_canvas_matplot_1.semilogy(energy_axis,self.spectra_dict[file_path]['sum_spec'],color=self.spectra_dict[file_path]['color'],label=self.spectra_dict[file_path]['label_short'])
                b,t = self.ax_canvas_matplot_1.get_ylim()
                l,r = self.ax_canvas_matplot_1.get_xlim()
                # set "home" view to new loaded data
#                if not self.win:    self.toolbar_matplot_1._views._elements[0] = [(l,r,b,t)]
        for file_path in self.spectra_dict:
            if list(self.spectra_dict[file_path]['sum_spec']) and self.spectra_dict[file_path]['plot'] and self.spectra_dict[file_path]['top']:
                energy_axis = list(map(ecalib,range(len(self.spectra_dict[file_path]['sum_spec']))))
                self.ax_canvas_matplot_1.semilogy(energy_axis,self.spectra_dict[file_path]['sum_spec'],color=self.spectra_dict[file_path]['color'],label=self.spectra_dict[file_path]['label_short'])
                self.ax_canvas_matplot_1.set_xlim(energy_axis[0],energy_axis[-1])
                self.ax_canvas_matplot_1.set_ylim(1,max(self.spectra_dict[file_path]['sum_spec']))
                b,t = self.ax_canvas_matplot_1.get_ylim()
                l,r = self.ax_canvas_matplot_1.get_xlim()
                # set "home" view to new loaded data
#                if not self.win:    self.toolbar_matplot_1._views._elements[0] = [(l,r,b,t)]
                for roi_label in self.spectra_dict[file_path]['gaussians']:
                    if self.spectra_dict[file_path]['gaussians'][roi_label]:
                        roi = self.spectra_dict[file_path]['rois'][roi_label]
                        energy_axis = list(map(ecalib,range(roi[0],roi[1])))
                        [[A,z0,x_0,FWHM],fit_dic] = self.spectra_dict[file_path]['gaussians'][roi_label]
                        [dA,dz0,dx_0,dFWHM] = np.sqrt(np.diag(fit_dic['cov']))
                        pos = ecalib(roi[0])+ecalib(x_0)
                        dpos = ecalib(dx_0)
                        width = ecalib(FWHM)-ecalib(0)
                        dwidth = ecalib(dFWHM)-ecalib(0)
                        resp = (roi[0]+x_0)/FWHM
                        dresp = resp*np.sqrt((dpos/pos)**2+(dwidth/width)**2)
                        self.ax_canvas_matplot_1.semilogy(energy_axis,fit_dic['fit'],color='red')
                        lines = []
                        lines.append(u'amplitude:\t{} \u00b1 {}'.format(*round_value_error(A,dA)))
                        lines.append(u'position:\t\t{} \u00b1 {}'.format(*round_value_error(pos,dpos)))
                        lines.append(u'FWHM:\t\t{} \u00b1 {}'.format(*round_value_error(width,dwidth)))
                        lines.append(u'resolving power:\t{} \u00b1 {}'.format(*round_value_error(resp,dresp)))
                        self.label_gaussian_fit.setText('\n'.join(lines))
        self.ax_canvas_matplot_1.legend(loc='upper right')
        
        # draw roi
        if len(self.spectra_dict) > 0:
            try:
                if not self.current_roi == 'all':
                    low = ecalib(self.spectra_dict[self.current_spectrum]['rois'][self.current_roi][0])
                    high = ecalib(self.spectra_dict[self.current_spectrum]['rois'][self.current_roi][1])
                    self.ax_canvas_matplot_1.axvspan(low, high, alpha=0.2, color = 'r')
                    self.ax_canvas_matplot_1.axvline(low, color = 'r')
                    self.ax_canvas_matplot_1.axvline(high, color = 'r')
            except:
                pass
        
        self.ax_canvas_matplot_1.set_title('Sum Spectrum')
        #try:
        if plotted_spectra > 0:
            self.ax_canvas_matplot_1.set_xlim([(ecalib(x1),ecalib(x2)) for x1,x2 in [x_limits]][0])
            self.ax_canvas_matplot_1.set_ylim(y_limits)
        #except:
        #    pass
        if self.ecalib_m == 1 and self.ecalib_n == 0:
            self.ax_canvas_matplot_1.set_xlabel('energy / counts')
        else:
            self.ax_canvas_matplot_1.set_xlabel('energy / keV')
        self.ax_canvas_matplot_1.set_ylabel('intensity / counts')
        self.canvas_matplot_1.draw()
#        self.label_roi_cursor = QtWidgets.QLabel('cursor = 0 cts')
#        self.label_roi_cursor.move(460,self.def_y2)
#        self.label_roi_cursor.setFixedHeight(self.def_height)
    
    
    def get_fp_1(self):
        el1 = str(self.entry_ecalib_el1.text())
        fl1 = str(self.entry_ecalib_fl1.text())
        try:    
            y1 = line_E(el1,fl1)
            self.entry_ecalib_E1.setText("{0:.3f}".format(y1))
        except: 
            self.entry_ecalib_E1.setText("no fp")
            print(ValueError('cannot load fp data'))  
    
    def get_fp_2(self):
        el2 = str(self.entry_ecalib_el2.text())
        fl2 = str(self.entry_ecalib_fl2.text())
        try:    
            y2 = line_E(el2,fl2)
            self.entry_ecalib_E2.setText("{0:.3f}".format(y2))
        except:
            self.entry_ecalib_E2.setText("no fp")
            print(ValueError('cannot load fp data'))        
    
    
    def calculate_calibration(self):
        try:
            x1 = float(self.entry_ecalib_ch1.text())
            x2 = float(self.entry_ecalib_ch2.text())
            y1 = float(self.entry_ecalib_E1.text())
            y2 = float(self.entry_ecalib_E2.text())
            
            m = (y2-y1)/(x2-x1)
            n = y2-m*x2
            self.entry_ecalib_m.setText("{0:.5f}".format(m))
            self.entry_ecalib_n.setText("{0:.5f}".format(n))
        except:
            print('cannot calibrate:', sys.exc_info()[0])
        
    
    def _fit_gaussian_to_ROI(self):
        try:
            data = self.spectra_dict[self.current_spectrum]['sum_spec']
            roi = self.spectra_dict[self.current_spectrum]['rois'][self.current_roi]
            param,fit_dic = fit_1d_gaussian(data[roi[0]:roi[1]],full_output=True)
            self.spectra_dict[self.current_spectrum]['gaussians'][self.current_roi] = [param,fit_dic]
            self._update_sum_spec_plot()
        except:
            print('cannot fit gaussian:', sys.exc_info()[0])
    
    
    def load_photon_maps(self, file_list, event_no, progress_callback):
        print('loading...')
        ### checks if the preview is loaded for the first time
        results = {}
        for file in file_list:
            for r in self.spectra_dict[file]['rois']:
                if not list(self.spectra_dict[file]['photon_maps'][r]): self.new_calc_pm = True
            if self.new_calc_pm:
                # for spe format
                if file.split('.')[-1] == 'spe':
                    results[file] = pt.make_photon_maps(file, self.spectra_dict[file]['rois'], event_no=event_no, progress_bar=progress_callback, n1=self.spectra_dict[file]['n1_n2'][0], n2=self.spectra_dict[file]['n1_n2'][1])
                    
                # for hdf5 format load cube if it is not available, yet -- make sure only one cube is loaded -- should be the one with current_spectrum
                elif file.split('.')[-1] in ['h5', 'hdf5', 'npy', 'npz']:
                    if self.spectra_dict[file]['cube'] is None:
                        cube, metadata = pt.load_cube(file)
                    else:
                        cube = self.spectra_dict[file]['cube']
                    results[file] = pt.get_pm_from_cube(cube, e_rois=self.spectra_dict[file]['rois'])
                    self.spectra_dict[file]['cube'] = None
                    if file == self.current_spectrum:
                        self.spectra_dict[file]['cube'] = cube
                ##########    
                self.new_calc_pm = False
            else:
                results[file] = self.spectra_dict[file]['photon_maps']
                self.new_calc_pm = False
        return results
    
        
    def _finish_load_PM(self, results):
        for file in results:
            self.spectra_dict[file]['photon_maps'] = results[file]
            for roi in self.spectra_dict[file]['rois']:
                combo_ind = self.comboBox_rois.findText(roi)
                if combo_ind == -1: combo_ind = self.comboBox_rois.findText(roi+' (calc)')
                self.comboBox_rois.setItemText(combo_ind, roi+' (calc)')
        self.progress.setValue(1)


    def _plot_PM(self):
        print("...loading completed!")
        time.sleep(0.1)
        self._update_photon_map()     ## evtl für alle in specs?
        
    def show_photon_map(self):
        do_all = bool(self.check_all_spectra.checkState())
        event_no = int(float(self.entry_event_no.text()))
        # try:
            # self.progress.setValue(1)
        if do_all:  specs = self.spectra_dict.keys()
        else:       specs = [self.current_spectrum]
    
        load_data_worker = Worker(self.load_photon_maps, specs, event_no)
        load_data_worker.signals.progress.connect(self._load_progress)
        load_data_worker.signals.result.connect(self._finish_load_PM)
        load_data_worker.signals.finished.connect(self._plot_PM)
        self.threadpool.start(load_data_worker)    
    
    
    
    
    
    
    
    
    # def show_photon_map(self):
    #     do_all = bool(self.check_all_spectra.checkState())
    #     try:
    #         # self.progress.setValue(1)
    #         if do_all:  specs = self.spectra_dict.keys()
    #         else:       specs = [self.current_spectrum]
            
    #         for spec in specs:
    #             ### checks if the preview is loaded for the first time
    #             for r in self.spectra_dict[spec]['rois']:
    #                 if not list(self.spectra_dict[spec]['photon_maps'][r]): self.new_calc_pm = True
    #             if self.new_calc_pm:
    #                 # self.spectra_dict[spec]['photon_maps'] = pt.make_photon_maps(spec, self.spectra_dict[spec]['rois'], self.progress, *self.spectra_dict[spec]['n1_n2'])
    #                 self.spectra_dict[spec]['photon_maps'] = pt.make_photon_maps(spec, self.spectra_dict[spec]['rois'], [], *self.spectra_dict[spec]['n1_n2'])
    #                 for roi in self.spectra_dict[spec]['rois']:
    #                     combo_ind = self.comboBox_rois.findText(roi)
    #                     if combo_ind == -1: combo_ind = self.comboBox_rois.findText(roi+' (calc)')
    #                     self.comboBox_rois.setItemText(combo_ind, roi+' (calc)')
    #                 self.new_calc_pm = False
    #             self._update_photon_map()
    #             # self.progress.setValue(0)
    #     except:
    #         print('cannot plot photon map:', sys.exc_info())
    
    
    def _update_photon_map(self):
        try:
            self.figure_matplot_2.delaxes(self.ax_canvas_matplot_2)
            self.pm_cb.remove()
        except:
            pass
        self.ax_canvas_matplot_2 = self.figure_matplot_2.add_subplot(111)
        self.ax_canvas_matplot_2.set_title('Photon Map')
        self.ax_canvas_matplot_2.set_xlabel('x axis')
        self.ax_canvas_matplot_2.set_ylabel('y axis')
        try:
            self.pm = self.ax_canvas_matplot_2.imshow(self.spectra_dict[self.current_spectrum]['photon_maps'][self.current_roi], interpolation ='nearest', origin='lower', cmap = plt.get_cmap('gray'))
        except:
            self.pm = self.ax_canvas_matplot_2.imshow(np.zeros((1024,1024)), interpolation ='nearest', origin='lower', cmap = plt.get_cmap('gray'))
        
        clim_low = float(self.entry_clim_low.text())
        clim_high = float(self.entry_clim_high.text())
        self.pm.set_clim(clim_low,clim_high)
        self.pm_cb = self.figure_matplot_2.colorbar(self.pm, ax=self.ax_canvas_matplot_2)
        self._update_image_profile_line()

        self.canvas_matplot_2.draw()
        self._update_GEXRF_plot()
    
    
    def _update_image_profile_line(self):
        try:
            x_max,y_max = np.shape(self.spectra_dict[self.current_spectrum]['photon_maps'][self.current_roi])
        except:
            x_max,y_max = 1024,1024
        phi = float(self.entry_profile_phi.text())
        if phi == 90:   phi = 90.0001               # otherwise might struggle with infinite values
        line = mt.image_line(phi,x_max,y_max)
        line_x = range(x_max)
        line_y = list(map(line,line_x))
        # restrict x and y
        restricted_line = [(x,y) for (x,y) in zip(line_x,line_y) if y > 0 if y < y_max if x > 0 if x < x_max]
        line_x = [x for (x,y) in restricted_line]
        line_y = [y for (x,y) in restricted_line]
        try:
            self.lineplot.pop(0).remove()
        except:
            pass
        self.lineplot = self.ax_canvas_matplot_2.plot(line_x,line_y,color='red')
        

    def save_photon_maps(self):
        try:
            do_all = bool(self.check_all_spectra.checkState())
            if do_all:  specs = self.spectra_dict.keys()
            else:       specs = [self.current_spectrum]
            r=1
            for spec in specs:
                for roi in self.spectra_dict[spec]['photon_maps']:
                    if list(self.spectra_dict[spec]['photon_maps']):
                        pt.save_photon_map(spec, roi, self.spectra_dict[spec]['photon_maps'][roi])
                    # self.progress.setValue(100.*r/(len(specs)*len(self.spectra_dict[spec]['photon_maps'])))
                    r+=1
            # self.progress.setValue(0)
        except:
            print('cannot save photon maps:', sys.exc_info())
        
        
    def add_roi(self):
        try:
            low = int(self.entry_roi_low.text())
            high = int(self.entry_roi_high.text())
            label = str(self.entry_roi_label.text())
            
            do_all = bool(self.check_all_spectra.checkState())
            if do_all:  specs = self.spectra_dict.keys()
            else:       specs = [self.current_spectrum]
            for spec in specs:
                self.spectra_dict[spec]['rois'][label]=[low,high]
                self.spectra_dict[spec]['photon_maps'][label]=[]
                self.spectra_dict[spec]['gaussians'][label]=[]
            self._update_roi_list(keep_top=label)
            self.new_calc_pm = True
        except:
            pass
    
        
    def save_rois(self):
        try:
            for spec in self.spectra_dict.keys():
                with open(spec[:-4]+'_roi.txt', 'w') as rf:
                    json.dump(self.spectra_dict[spec]['rois'], rf)
        except:
            pass

    def load_rois(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getOpenFileName(self, 'open file', self.current_spectrum, "roi files (*.txt)")    ### function for a folder-GUI
        try:
            data_path, types = output
        except:
            data_path = output
        if not data_path == '':
            self.entry_current_file.setText(str(data_path))
        try:
            do_all = bool(self.check_all_spectra.checkState())
            if do_all:  specs = self.spectra_dict.keys()
            else:       specs = [self.current_spectrum]
            for spec in specs:
                with open(data_path, 'r') as rf:
                    roi_dic = json.load(rf)
                for roi in roi_dic.keys():
                    self.spectra_dict[spec]['rois'][roi]=roi_dic[roi]
                    self.spectra_dict[spec]['photon_maps'][roi]=[]
                    self.spectra_dict[spec]['gaussians'][roi]=[]
            self._update_roi_list(keep_top=roi)
            self.new_calc_pm = True
        except:
            pass
    
    def del_roi(self):
        try:
            do_all = bool(self.check_all_spectra.checkState())
            if do_all:  specs = self.spectra_dict.keys()
            else:       specs = [self.current_spectrum]
            
            for spec in specs:
                self.spectra_dict[spec]['rois'].pop(self.current_roi)
                self.spectra_dict[spec]['photon_maps'].pop(self.current_roi)
                self.spectra_dict[spec]['gaussians'].pop(self.current_roi)
        except:
            print('rois list is already empty')
        self._update_roi_list()       
    
    
    def _select_roi(self,label):
        roi = str(label).replace(' (calc)','')
        self.current_roi = roi
        self._update_photon_map()
        try:
            [l,u] = self.spectra_dict[self.current_spectrum]['rois'][self.current_roi]
            self.label_current_roi.setText('('+str(l)+','+str(u)+')')
        except:
            pass
        if list(self.spectra_dict[self.current_spectrum]['sum_spec']):
            self._update_sum_spec_plot()
    
    
    def _update_roi_list(self, keep_top='you_chose_a_weird_name_for_a_roi'):
        self.comboBox_rois.clear()
        roi_list = []
        if len(self.spectra_dict) > 0:              # only, if not all elements are deleted
            for roi in self.spectra_dict[self.current_spectrum]['rois']:
                if list(self.spectra_dict[self.current_spectrum]['photon_maps'][roi]):
                    roi_list.append(roi+' (calc)')
                else:
                    roi_list.append(roi)
            self.comboBox_rois.addItems(roi_list)
            combo_ind = self.comboBox_rois.findText(keep_top)
            if not combo_ind == -1:
                self.comboBox_rois.setCurrentIndex(combo_ind)
            self._select_roi(self.comboBox_rois.currentText())                  # triggers _update_sum_spec_plot
        else:
            self._update_sum_spec_plot()
        self._update_photon_map()


    def save_profile(self):
        if list(self.current_gexrf_profile):
            dialog = QtWidgets.QFileDialog(self)
            output = dialog.getSaveFileName(self, 'save file', self.current_spectrum)    ### function for a folder-GUI
            try:
                data_path, types = output
            except:
                data_path = output
            if not data_path == '':
                np.savetxt(str(data_path),self.current_gexrf_profile)
        
        
    def _update_GEXRF_plot(self):

        try:
            pm = self.spectra_dict[self.current_spectrum]['photon_maps'][self.current_roi]
        except:
            pm = []
        try:
            phi = float(self.entry_profile_phi.text())
        except:
            phi = 0
        self.ax_canvas_matplot_3.clear()
        if list(pm):
            pm_tmp = scipy.ndimage.interpolation.rotate(pm,phi,cval=-1)
            x_axis = range(np.shape(pm_tmp)[1])
            
            pm = np.array(pm_tmp)
            pm[pm<0] = 0
            pm_tmp[pm_tmp>0] = 0
            
            y_axis = np.sum(pm,axis=0)
            norm_fac = np.shape(pm_tmp)[1]+np.sum(pm_tmp,axis=0)      # summiert alle -1 auf -> gibt Wert fuer Anzahl der Pixel mit tatsaechlichen Werten
            norm_fac /= np.max(norm_fac)
            y_axis /= norm_fac
            self.current_gexrf_profile = np.array([x_axis,y_axis]).T
            self.ax_canvas_matplot_3.plot(x_axis,y_axis,color=self.spectra_dict[self.current_spectrum]['color'])

        self.ax_canvas_matplot_3.set_title('Profile')
        self.ax_canvas_matplot_3.set_xlabel('column / pixel')
        self.ax_canvas_matplot_3.set_ylabel('norm intensity')
        self.canvas_matplot_3.draw()
        self.figure_matplot_3.tight_layout()
    
    def _get_n1_n2(self):
        n_min = int(self.entry_event_n_min.text())
        n_max = int(self.entry_event_n_max.text())
        return n_min,n_max
    
    def _set_n1_n2(self,n1,n2):
        self.entry_event_n_min.setText('{:.0f}'.format(n1))
        self.entry_event_n_max.setText('{:.0f}'.format(n2))
        
    
            
            
### define functions used by plot interaction ###
    def _on_button_press(self, event):
        self.x_press_event = event.xdata
    
    def _on_button_release(self, event):
        x = event.xdata
        adu_low = '0'
        adu_high = '0'
        kev_low = ""
        try:
            x_low, x_high = np.sort([self.x_press_event, x])
            m,n = float(self.entry_ecalib_m.text()),float(self.entry_ecalib_n.text())
            inv_ecal = lambda y: (y-n)/m
            adu_low = "{0:.0f}".format(inv_ecal(x_low))
            adu_high = "{0:.0f}".format(inv_ecal(x_high))
            if m==1 and n==0:
                kev_low = ""
            else:
                kev_low = ", {0:.3f} keV".format(x_low)
        except:
            print(self.entry_ecalib_n.text(), type(self.entry_ecalib_n.text()))
            print(self.entry_ecalib_m.text(), type(self.entry_ecalib_m.text()))

        if adu_low == adu_high:
            self.label_roi_cursor.setText('cursor @ '+adu_low+' ADU'+kev_low)
        else:
            self.entry_roi_low.setText(adu_low)
            self.entry_roi_high.setText(adu_high)
        
        
### define functions used by PM plot interaction ###
    def _on_button_press_PM(self, event):
        self.x_press_event_PM = int(event.xdata)
        self.y_press_event_PM = int(event.ydata)
    
    def _on_button_release_PM(self, event):
        x = int(event.xdata)
        y = int(event.ydata)
        
        x1 = min([self.x_press_event_PM,x])
        x2 = max([self.x_press_event_PM,x])
        y1 = min([self.y_press_event_PM,y])
        y2 = max([self.y_press_event_PM,y])
        self.entry_event_x_min.setText(str(x1))
        self.entry_event_x_max.setText(str(x2))
        self.entry_event_y_min.setText(str(y1))
        self.entry_event_y_max.setText(str(y2))
        
        try:
            self.rect_inner.remove()
            self.rect_outer.remove()
        except:
            pass
        self.rect_inner = patches.Rectangle((x1,y1), x2-x1, y2-y1, facecolor='white', alpha=0.1)
        self.rect_outer = patches.Rectangle((x1,y1), x2-x1, y2-y1, facecolor='None', edgecolor='blue', alpha=1)
        if not self.rect_inner is None or not self.rect_outer is None:
            self.ax_canvas_matplot_2.add_patch(self.rect_inner)
            self.ax_canvas_matplot_2.add_patch(self.rect_outer)
            self.canvas_matplot_2.draw()
        
        self._plot_xyroi_spec()

    def _plot_xyroi_spec(self):
        E_min = int(self.entry_event_E_min.text())
        E_max = int(self.entry_event_E_max.text())
        # xy_roi = [[int(self.entry_event_x_min.text()), int(self.entry_event_y_min.text())],
        #           [int(self.entry_event_x_max.text()), int(self.entry_event_y_max.text())]]
        xy_roi = [[int(self.entry_event_y_min.text()), int(self.entry_event_y_max.text())],
                  [int(self.entry_event_x_min.text()), int(self.entry_event_x_max.text())]]

        if self.current_spectrum.split('.')[-1] == 'spe':
            sum_spec, n_spec, mode = pt.get_spectrum(self.current_spectrum, 10000, Emin_Emax=(E_min,E_max), xy_roi = xy_roi, progress_bar=self.progress)
            sum_spec = np.sum(n_spec,axis=0)
        elif self.current_spectrum.split('.')[-1] in ['h5', 'hdf5', 'npy', 'npz']:
            ### first erase all loaded cubes but from current spectrum
            for dp in self.spectra_dict:
                if dp != self.current_spectrum:
                    self.spectra_dict[dp]['cube'] = None
            if self.spectra_dict[self.current_spectrum]['cube'] is None:
                cube, metadata = pt.load_cube(self.current_spectrum)
                self.spectra_dict[self.current_spectrum]['cube'] = cube  
            else:
                cube = self.spectra_dict[self.current_spectrum]['cube']
            sum_spec = pt.get_spectrum_from_cube(cube, e_roi=(E_min,E_max), xy_roi=xy_roi)#[[xy_roi[0][0], xy_roi[1][0]], [xy_roi[0][1], xy_roi[1][1]]])
            
        # self._select_spectrum()                              # triggers update_sum_spec_plot
        self._update_sum_spec_plot()

        self.progress.setValue(0)

        uncalib = lambda x: (x-self.ecalib_n)/self.ecalib_m
        x_limits=list(map(uncalib,self.ax_canvas_matplot_1.get_xlim()))
        y_limits=self.ax_canvas_matplot_1.get_ylim()
        self.ecalib_m = float(self.entry_ecalib_m.text())
        self.ecalib_n = float(self.entry_ecalib_n.text())
        ecalib = lambda x: self.ecalib_m * x + self.ecalib_n
        
        self.ax_canvas_matplot_1.set_xlim(x_limits)
        self.ax_canvas_matplot_1.set_ylim(y_limits)
        
        energy_axis = list(map(ecalib,range(len(sum_spec))))
        try:
            self.ax_canvas_matplot_1.semilogy(energy_axis,sum_spec/np.max(sum_spec)*np.max(self.spectra_dict[self.current_spectrum]['sum_spec']),color='red',linestyle=':')
            # self.ax_canvas_matplot_1.set_xlim(energy_axis[0],energy_axis[-1])
            # self.ax_canvas_matplot_1.set_ylim(1,max(self.spectra_dict[self.current_spectrum]['sum_spec']))
        except: # no  spectrum plotted yet
            self.ax_canvas_matplot_1.semilogy(energy_axis,sum_spec/np.max(sum_spec),color='red',linestyle=':')
            # self.ax_canvas_matplot_1.set_xlim(energy_axis[0],energy_axis[-1])
            # self.ax_canvas_matplot_1.set_ylim(1,max(sum_spec))
        
        # b,t = self.ax_canvas_matplot_1.get_ylim()
        self.canvas_matplot_1.draw()
            
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()
    EXA_GUI = evares_GUI_main()
    EXA_GUI.show()
    app.exec_()

if __name__ == '__main__':
    main()
    
    
#J:/TUB/Projekte/TubeGEXRF/VM/Data/2018/20180621_NiCu/NiCu_9mu_230muA_500ms.spe
