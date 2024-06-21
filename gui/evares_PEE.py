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
import ntpath
import inspect
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.backends.backend_qt5agg as pltqt
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

import sys
sys.path.append('../')


from base import misc_tools as mt
from base import pee_tools as pt
from base.pee_module import pee_class

from mpl_toolkits.mplot3d import Axes3D
import time


def data_path():
    """
    returns the path where the evares files are installed.

    Author: JB
    """

    data_path_list = os.path.abspath(inspect.getfile(inspect.currentframe())).split(os.path.sep)[:-1]
    data_path = ''
    for element in data_path_list:
        data_path += element + '/'
    return data_path

Ui_MainWindow, QMainWindow = loadUiType(data_path()+'evares_PEE.ui')

class PanOnlyToolbar(pltqt.NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in pltqt.NavigationToolbar2QT.toolitems if
                 t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save", )]

#    def __init__(self, *args, **kwargs):
#        super(PanOnlyToolbar, self).__init__(*args, **kwargs)
#        self.layout().takeAt(1)  #or more than 1 if you have more buttons


class pee_GUI_main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        if platform.system() == 'Windows':  self.win = True
        else:                               self.win = False
        if not self.win:
            sp.call('clear',shell = True) # clear the screen to get rid of all the print
        self.sum_spec = []
        
        ### define geometry and properties ###   
        plt.rcParams['savefig.dpi'] = 300
        self.setWindowTitle('pee')
        
        
        self.def_button_pressed = 2
        self.last_path = ''
        
        self.md_name = ''
        self.MD_median_frame = np.zeros((1024,1024))
        self.MD_mean_frame = np.zeros((1024,1024))
        self.MD_std_frame = np.zeros((1024,1024))
        self.defect_map = None
        self.dark_file_mean_arr = []
        
        self.current_roi = 'all'
        self.new_calc_pm = True
        self.spectra_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.store_mode = self.button_store_mode.text()
        
        self.pee_modes = {}
        self.pee_modes['4px-Area']='four_px_area'
        self.pee_modes['Clustering']='clustering'
        self.pee_modes['4px-Area-Clustering']='four_px_area_clustering'
        self.pee_modes['Gaussian-Model-Fit (no jit)']='gaussian_model_fit'
        self.pee_modes['Quick Gaussian-Model-Fit']='qgmf'
        self.pee_modes['ASCA']='gendreau'
        self.pee_modes['EPIC']='epic'
        try:
            import torch
        except:
            print('pytorch not detected')
        else:
            self.pee_modes['Small UNet']='small_unet'
            self.pee_modes['Big UNet']='big_unet'
        
        self.evaluation = pee_class('', '', '', '', '')
        self.evaluation.emitter.send_metadata.connect(self.print_signal)
        self.time_per_event_list = []
        self.time_per_frame_list = []
        
        self.__init_UI()


    def __init_UI(self):  
        ''' define the layout of the UI '''
        ### set actions connected to the menu or toolbar ###
        
        ### set StatusBar ###
        self.statusBar()
        
        ### set functions to buttons ###
        
        self.button_get_MD_filenames.clicked.connect(self._get_dark_filename)
        self.button_calc_MD.clicked.connect(self._calc_MD)
        self.button_sim_MD.clicked.connect(self._sim_MD)
        self.slider_white.valueChanged.connect(self._update_MD_plot)
        self.slider_black.valueChanged.connect(self._update_MD_plot)
        self.button_load_MD.clicked.connect(self._load_MD)
        self.button_save_MD.clicked.connect(self._save_MD)
        self.button_kill_MD.clicked.connect(self._kill_MD)
        self.button_load_dm.clicked.connect(self._load_dm)
        self.button_make_pee.clicked.connect(self._pee)
        self.button_store_mode.clicked.connect(self._change_storemode)
        
        self.button_get_data_filenames.clicked.connect(self._get_data_filename)
        self.button_get_eval_dir.clicked.connect(self._get_eval_dir)
        self.button_get_eval_name.clicked.connect(self._get_eval_name)
        
        self.button_update_pee_stats.clicked.connect(self._update_pee_stats)
        self.button_save_npx_specs.clicked.connect(self._save_npx_specs)
        
        
        ### set pee modes in comboBox_pee_mode
        
        for mode in self.pee_modes:
            self.comboBox_pee_mode.addItem(mode)
        mode_ind = self.comboBox_pee_mode.findText("Clustering", Qt.MatchFixedString)
        if mode_ind >= 0:   self.comboBox_pee_mode.setCurrentIndex(mode_ind)
        self.comboBox_pee_mode.activated[str].connect(self._change_pee_mode)
        self.comboBox_pee_dark.activated[str].connect(self._change_dark_mode)
        
        ### set functions to comboBoxes ###
        
        self.comboBox_dark_frame.activated[str].connect(self._update_MD_plot_comboBox)
        
        ### init canvas
        fig = Figure()
        self.addmpl_dark(fig)
        self.addmpl_eval(fig)


    def addmpl_dark(self, fig):
        self.canvas_dark = FigureCanvas(fig)
        self.mplwin_dark_vl.addWidget(self.canvas_dark)
        self.canvas_dark.draw()
        self.toolbar_dark = NavigationToolbar(self.canvas_dark, self.mplwin_dark, coordinates=True)
        self.mplwin_dark_vl.addWidget(self.toolbar_dark)

    def addmpl_eval(self, fig):
        self.canvas_eval = FigureCanvas(fig)
        self.mplwin_eval_vl.addWidget(self.canvas_eval)
        self.canvas_eval.draw()
        self.toolbar_eval = NavigationToolbar(self.canvas_eval, self.mplwin_eval, coordinates=True)
        self.mplwin_eval_vl.addWidget(self.toolbar_eval)

    def rm_dark(self):
        self.mplwin_dark_vl.removeWidget(self.canvas_dark)
        self.canvas_dark.close()
        self.mplwin_dark_vl.removeWidget(self.toolbar_dark)
        self.toolbar_dark.close()
        
    def rm_eval(self):
        self.mplwin_eval_vl.removeWidget(self.canvas_eval)
        self.canvas_eval.close()
        self.mplwin_eval_vl.removeWidget(self.toolbar_eval)
        self.toolbar_eval.close()
        
        
    def print_signal(self, status):
        
        self.progress.setValue(int(100.*status["frame number"]/status["frames todo"]))
        
        photon_num = status['photon number']
        eval_time = status['evaluation time']
        
        time_per_event = eval_time/photon_num
        self.time_per_event_list.append(time_per_event)
        self.time_per_frame_list.append(eval_time)
        
        eff_str = 'current frame: '+str(photon_num)+' ph\n '
        eff_str += str(round(time_per_event*1000,2))+' ms/ph, '
        eff_str += str(round(eval_time,2))+' s/im\n'
        eff_str += 'mean:\n'+str(round(np.mean(self.time_per_event_list)*1000,2))+' ms/ph, '
        eff_str += str(round(np.mean(self.time_per_frame_list),2))+' s/im'
        self.label_efficiency_meas.setText(eff_str)

        
    def _get_dark_filename(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getOpenFileName(self, 'open file', self.last_path, "image files (*.txt *.dat *.tsv *.tif *.tiff *.TIF *.TIFF *.spe *.SPE)")    ### function for a folder-GUI
        try:    data_path, types = output
        except: data_path = output
        if not data_path == '':
            self.last_path = data_path
            dp = data_path.replace('\\','/')
            dp = '/'.join(data_path.split('/')[:-1])+'/*.'+dp.split('.')[-1]
            self.entry_current_file.setText(dp)
        
    def _get_data_filename(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getOpenFileName(self, 'open file', self.last_path, "image files (*.txt *.dat *.tsv *.tif *.tiff *.TIF *.TIFF *.spe *.SPE)")    ### function for a folder-GUI
        try:    data_path, types = output
        except: data_path = output
        if not data_path == '':
            self.last_path = data_path
            dp = data_path.replace('\\','/')
            dp = '/'.join(data_path.split('/')[:-1])+'/*.'+dp.split('.')[-1]
            self.entry_path_data.setText(dp)
            self.entry_path_eval.setText('/'.join(data_path.split('/')[:-2]))
            self.entry_eval_name.setText(data_path.split('/')[-2])
    
    def _get_eval_dir(self):
        dialog = QtWidgets.QFileDialog(self)
        output = str(dialog.getExistingDirectory(self, 'open file', self.last_path))    ### function for a folder-GUI
        try:    data_path, types = output
        except: data_path = output
        if not data_path == '':
            self.last_path = data_path
            self.entry_path_eval.setText(data_path)
        
    def _get_eval_name(self):
        dialog = QtWidgets.QFileDialog(self)
        output = dialog.getOpenFileName(self, 'open file', self.last_path, 'pee files (*.pee *.h5 *.hdf5)')    ### function for a folder-GUI
        try:    data_path, types = output
        except: data_path = output
        if not data_path == '':
            self.last_path = data_path
            self.entry_eval_name.setText(data_path.replace('\\','/').split('/')[-1])
            self.entry_path_eval.setText(os.path.dirname(data_path))
        
    
    def _calc_MD(self):

        data_path = str(self.entry_current_file.text())
        data_path = data_path.replace('\\','/')
        image_files = ['txt','dat','tsv','tif', 'tiff', 'TXT', 'DAT', 'TSV', 'TIF', 'TIFF', 'spe', 'SPE']
        if os.path.isdir(data_path): 
            if data_path[-1] == '/':    data_paths = [data_path+'*.'+imfi for imfi in image_files]
            else:                       data_paths = [data_path+'/*.'+imfi for imfi in image_files]
            data_path_list = [glob.glob(dps) for dps in data_paths]
        else:
            data_path_list = glob.glob(data_path)
        if self.entry_data_num.text().lower() in ['all', 'inf', '-1', 'oo']: data_num = len(data_path_list)
        else:                                   data_num = int(self.entry_data_num.text())
        data_path_list = list(mt.sort_files(data_path_list, mode='created')[0])[:data_num]
        print(f'Calculating master dark. {data_num} images will be used...')
        md_mode = str(self.comboBox_dark_mode.currentText())
        parts = int(self.entry_MD_parts.text())
        limit = float(self.entry_MD_limit.text())
        self.MD_median_frame, self.MD_std_frame, tmp, self.dark_file_mean_arr, self.MD_mean_frame = mt.std_pic('', file_list=data_path_list, parts=parts, limit=limit, progress_bar=self.progress)
        self.progress.setValue(100)
        self.MD_mean_frame = np.array(self.MD_mean_frame, dtype='float64')
#        print(self.MD_mean_frame
#        print(np.dtype(self.MD_mean_frame)
        self.MD_std = np.mean(self.MD_std_frame)
        
        if md_mode == 'global noise':
            sig_arr = []
            for f in data_path_list:
                sig_arr.append(np.std(mt.load_image_file(f)))
            self.MD_std_frame.fill(np.mean(sig_arr))
        
        self._update_MD_plot_comboBox()
        self.progress.setValue(0)
        

    def _sim_MD(self):
        offset = float(self.entry_offset.text())
        std = float(self.entry_sig.text())
        MD_x = int(self.mdsim_x.text())
        MD_y = int(self.mdsim_y.text())
        self.MD_mean_frame = np.full((MD_x, MD_y), offset)
        self.MD_median_frame = self.MD_mean_frame.astype(int)
        self.MD_std_frame = np.random.normal(0, std, (MD_x, MD_y))
        self.progress.setValue(100)
        self.MD_std = np.mean(self.MD_std_frame)
        self._update_MD_plot_comboBox()
        self.progress.setValue(0)        
        
        
    def _load_MD(self):
        try:
            dp = str(self.entry_current_file.text())
            dp = dp.replace('\\','/')
            dialog = QtWidgets.QFileDialog(self)
            output = dialog.getOpenFileName(self, 'open file', dp, "numpy container (*.npz *.npy);; image files (*.txt *.dat *.tsv *.tif *.tiff)")    ### function for a folder-GUI
            try:    data_path, types = output
            except: data_path = output
            if not data_path == '':
                self.md_name = ntpath.basename(data_path)
                try:
                    if data_path.split('.')[-1] == 'npy':
                        [self.MD_median_frame, self.MD_mean_frame,self.MD_std_frame,self.dark_file_mean_arr] = np.load(data_path)
                    elif data_path.split('.')[-1] == 'npz':
                        npz_dic = np.load(data_path)
                        self.MD_median_frame = npz_dic['a']
                        self.MD_mean_frame = npz_dic['b']
                        self.MD_std_frame = npz_dic['c']
                        self.dark_file_mean_arr = npz_dic['d']
                    else:
                        raise
                except:
                    print('load master dark: no numpy container found')
                    dtype = data_path.split('.')[-1]
                    try:
                        self.MD_median_frame = mt.load_image_file('_'.join(data_path.split('_')[:-1])+'_median.'+dtype)
                        self.MD_mean_frame = mt.load_image_file('_'.join(data_path.split('_')[:-1])+'_mean.'+dtype)
                        self.MD_std_frame = mt.load_image_file('_'.join(data_path.split('_')[:-1])+'_std.'+dtype)
                    except:
                        print('load master dark: no _median, _mean and _std frame found')
                        try:
                            self.MD_mean_frame = mt.load_image_file(data_path)
                        except:
                            print('IOError: could not load data for Master Dark frame. Please use the correct .npy or .tif file')
                        else:
                            print('single dark frame loaded, standard deviation calculated')
                            self.MD_median_frame = self.MD_mean_frame
                            self.MD_std_frame = np.ones(np.shape(self.MD_median_frame))*np.std(self.MD_median_frame)
                            
                self._update_MD_plot_comboBox()
        except:
            pass
        
        
    def _save_MD(self):
        try:
            dp = str(self.entry_current_file.text())
            dp = dp.replace('\\','/')
            dialog = QtWidgets.QFileDialog(self)
            output = dialog.getSaveFileName(self, 'save file', '/'.join(dp.split('/')[:-1]), "as numpy container (*.npz);;as image (*.tif *.txt *.tsv *.dat)")
            try:    data_path, types = output
            except: data_path = output
            if not data_path == '':
                # print(data_path)
                dtype = data_path.split('.')[-1]
                # print(dtype)
                # print('data type of arrays: ', self.MD_median_frame.dtype)
                if types == 'as numpy container':
                    np.savez_compressed(data_path, a=self.MD_median_frame.astype('uint16'), b=self.MD_mean_frame.astype('float32'), c=self.MD_std_frame.astype('float32'), d=self.dark_file_mean_arr)
                else:
                    if list(self.MD_std_frame) and list(self.MD_median_frame) and list(self.MD_mean_frame):
                        pure_name = '.'.join(data_path.split('.')[:-1])
                        print(pure_name)
                        if dtype == 'txt' or dtype == 'dat' or dtype == 'tsv':
                            np.savetxt(pure_name+'_median.'+dtype,self.MD_median_frame)
                            np.savetxt(pure_name+'_mean.'+dtype,self.MD_mean_frame)
                            np.savetxt(pure_name+'_std.'+dtype,self.MD_std_frame)
                        elif dtype == 'tif':
                            mt.save_image_file(pure_name+'_median.'+dtype, self.MD_median_frame,'uint16')
                            mt.save_image_file(pure_name+'_mean.'+dtype, self.MD_mean_frame,'float32')
                            mt.save_image_file(pure_name+'_std.'+dtype, self.MD_std_frame,'float32')
                        elif dtype == 'npz':
    #                        np.savez_compressed(data_path, a=self.MD_median_frame, b=self.MD_mean_frame, c=self.MD_std_frame, d=self.dark_file_mean_arr)
                            np.savez_compressed(data_path, a=self.MD_median_frame.astype('uint16'), b=self.MD_mean_frame.astype('float32'), c=self.MD_std_frame.astype('float32'), d=self.dark_file_mean_arr)
                        else:
                            mt.save_image_file(pure_name+'_median.'+dtype, self.MD_median_frame,'uint16')
                            mt.save_image_file(pure_name+'_mean.'+dtype, self.MD_mean_frame,'float32')
                            mt.save_image_file(pure_name+'_std.'+dtype, self.MD_std_frame,'float32')
                    else:
                        print('IOError: no master dark frame calculated, yet')
        except:
            print(sys.exc_info(), '\nIOError: master dark frame could not be saved!\n')


    def _kill_MD(self):
        self.MD_median_frame = []
        self.MD_mean_frame = []
        self.MD_std_frame = []
        self._update_MD_plot_comboBox()
    
    
    def _load_dm(self):
        try:
            dp = str(self.entry_current_file.text())
            dp = dp.replace('\\','/')
            dialog = QtWidgets.QFileDialog(self)
            output = dialog.getOpenFileName(self, 'open file', dp, "image files (*.png *.tif *.txt *.dat *.tsv)")    ### function for a folder-GUI
            try:    data_path, types = output
            except: data_path = output
            if not data_path == '':
                defect_map = mt.load_image_file(data_path)
                print(np.shape(defect_map))
                if len(np.shape(defect_map)) == 3:
                    if np.shape(defect_map)[2] == 4 or np.shape(defect_map)[2] == 3:    # image is rgb
                        try:
                            self.defect_map = defect_map[:,:,3]     # mit Alpha Kanal
                        except:
                            self.defect_map = defect_map[:,:,2]     # ohne Alpha Kanal
                            tmp_map = np.array(self.defect_map)
                            self.defect_map[tmp_map==0]=255
                            self.defect_map[tmp_map==255]=0
                    else:
                        self.defect_map = defect_map[:,:,np.shape(defect_map)[3]-1]
                elif len(np.shape(defect_map)) == 2:
                    self.defect_map = defect_map
                else:
                    self.defect_map = None
                print('defect map corner:\n',self.defect_map[0:5,0:5])
                print('defect map shape:', np.shape(self.defect_map))
        except:
            pass
        
    def _update_MD_plot_comboBox(self):
        self.slider_white.setValue(20)
        self.slider_black.setValue(80)
        self._update_MD_plot()
        

    def _update_MD_plot(self):
        md_figure = Figure()
        ax1 = md_figure.add_subplot(111)
        
        md_frame = str(self.comboBox_dark_frame.currentText())
        ax1.set_title('Master Dark (MD)')
        if list(self.MD_median_frame) and 'pixelwise median' in md_frame:           data = self.MD_median_frame
        elif list(self.MD_mean_frame) and 'pixelwise mean' in md_frame:             data = self.MD_mean_frame
        elif list(self.MD_std_frame) and 'pixelwise noise (rms)' in md_frame:       data = self.MD_std_frame
        elif list(self.dark_file_mean_arr) and 'dark frame evolution' in md_frame:  
            data = np.array(self.dark_file_mean_arr)-np.mean(self.MD_mean_frame)
        elif (not self.defect_map is None) and ('defect map' in md_frame):          data = self.defect_map
        else:                                                                       data = np.zeros((1024,1024))
        if 'histogram' in md_frame:
            try:
                md = ax1.hist(np.ravel(data), 'auto', histtype='step')
            except:#np.arange(data.min(),data.max(),0.1)
                md = ax1.hist([0,0], np.arange(0,1,0.1), histtype='step')
            ax1.set_xlabel('intensity / ADU')
            ax1.set_ylabel('number')
        elif 'dark frame evolution' in md_frame:
            try:
                md = ax1.plot(data, 'bo')
            except:#np.arange(data.min(),data.max(),0.1)
                md = ax1.plot([], 'bo')
            ax1.set_ylabel('deviation from mean / ADU')
            ax1.set_xlabel('number')
        else:
            try:
                md = ax1.imshow(data, interpolation ='nearest', origin='lower', cmap = plt.get_cmap('gray'))
            except:
                md = ax1.imshow(np.zeros((1024,1024)), interpolation ='nearest', origin='lower', cmap = plt.get_cmap('gray'))
            ax1.set_xlabel('x axis')
            ax1.set_ylabel('y axis')
            
            clim_low = float(self.slider_white.value())*0.01*min(np.max(data), 10*np.median(data))
            clim_high = float(self.slider_black.value())*0.01*min(np.max(data), 10*np.median(data))
            
            md.set_clim(clim_low,clim_high)
            md_cb = md_figure.colorbar(md, ax=ax1)

        self.rm_dark()
        self.addmpl_dark(md_figure)
        self._update_MD_stats()
    
    
    def _update_MD_stats(self):
        md_frame = str(self.comboBox_dark_frame.currentText())
        if 'pixelwise median' in md_frame:   current_frame = self.MD_median_frame
        elif 'pixelwise mean' in md_frame:   current_frame = self.MD_mean_frame
        elif 'pixelwise noise (rms)' in md_frame:   current_frame = self.MD_std_frame
        elif 'dark frame evolution' in md_frame:    current_frame = np.array(self.dark_file_mean_arr)-np.mean(self.MD_mean_frame)
        else:   current_frame = [0]
        if not list(current_frame):
            current_frame = [0]
        MD_min = np.min(current_frame)
        min_where = np.where(current_frame == MD_min)
        MD_max = np.max(current_frame)
        max_where = np.where(current_frame == MD_max)
        try:
            MD_min_pos = (min_where[1][0],min_where[0][0])
            MD_max_pos = (max_where[1][0],max_where[0][0])
        except:
            MD_min_pos = (0,min_where[0][0])
            MD_max_pos = (0,max_where[0][0])
        
        MD_mean = np.mean(current_frame)
        MD_std = np.std(current_frame)
        
        lines = []
        lines.append('MD name: {}'.format(self.md_name))
        lines.append('stats:\tmin:\t{:.1f}\t({}, {})'.format(MD_min,*MD_min_pos))
        lines.append('\tmax:\t{:.1f}\t({}, {})'.format(MD_max,*MD_max_pos))
        lines.append('\tmean:\t{:.1f}'.format(MD_mean))
        lines.append('\tstd:\t{:.2f}'.format(MD_std))
        self.label_MD_stats.setText('\n'.join(lines))
        

    def _change_pee_mode(self):
        mode = self.pee_modes[str(self.comboBox_pee_mode.currentText())]
        print(mode)
        if mode in ['four_px_area', 'gendreau', 'epic']:
            self.label_T1.setEnabled(True)
            self.entry_sig_fac_1.setEnabled(True)
            self.label_T2.setEnabled(False)
            self.label_T2.setText('T2:')
            self.label_T2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setEnabled(False)
            self.label_fitarea.setEnabled(False)
            self.label_fitarea.setText('fit area:')
            self.label_fitarea.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setEnabled(False)
        elif mode in ['clustering', 'four_px_area_clustering']:
            self.label_T1.setEnabled(True)
            self.entry_sig_fac_1.setEnabled(True)
            self.label_T2.setEnabled(True)
            self.label_T2.setText('T2:')
            self.label_T2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setEnabled(True)
            self.label_fitarea.setEnabled(False)
            self.label_fitarea.setText('fit area:')
            self.label_fitarea.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setEnabled(False)
        elif mode in ['gaussian_model_fit', 'qgmf']:
            self.label_T1.setEnabled(True)
            self.entry_sig_fac_1.setEnabled(True)
            self.label_T2.setEnabled(False)
            self.label_T2.setText('T2:')
            self.label_T2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setToolTip('<html><head/><body><p>Like noise threshold 1 but used for pixels surrounding\the event center in pee-modes &quot;clustering&quot; and &quot;4px-Area-Clustering&quot;.</p></body></html>')
            self.entry_sig_fac_2.setEnabled(False)
            self.label_fitarea.setEnabled(True)
            self.label_fitarea.setText('fit area:')
            self.label_fitarea.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setToolTip('<html><head/><body><p>Defines the pixel number of the square edge used for\&quot;gaussian model fit&quot; pee-mode. For odd numbers the\maximum is in the middle pixel of the square</p></body></html>')
            self.entry_squ_a.setEnabled(True)
        elif mode in ['small_unet', 'big_unet']:
            self.label_T1.setEnabled(True)
            self.entry_sig_fac_1.setEnabled(True)
            self.label_T2.setEnabled(True)
            self.label_T2.setText('parts:')
            self.label_T2.setToolTip('divide image in parts x parts subimages')
            self.entry_sig_fac_2.setToolTip('divide image in parts x parts subimages')
            self.entry_sig_fac_2.setEnabled(True)
            self.entry_sig_fac_2.setText('1')
            self.label_fitarea.setEnabled(True)
            self.label_fitarea.setText('ADU max')
            self.label_fitarea.setToolTip('estimate of maximum image signal necessary to normalize for networks')
            self.entry_squ_a.setToolTip('estimate of maximum image signal necessary to normalize for networks')
            self.entry_squ_a.setEnabled(True)
            
    
    
    def _change_dark_mode(self):
        mode = str(self.comboBox_pee_dark.currentText())
        print(mode)
        if mode == 'Rolling Dark':
            self.label_num_rol_dark.setEnabled(True)
            self.entry_rolling_dark_no.setEnabled(True)
        else:
            self.label_num_rol_dark.setEnabled(False)
            self.entry_rolling_dark_no.setEnabled(False) 
            

    def _change_storemode(self):
        if self.store_mode == '.pee':
            self.button_store_mode.setText('.h5')
            self.store_mode = '.h5'
            self.h5 = True
            self.check_pee_red.setEnabled(False)
        else:
            self.button_store_mode.setText('.pee')
            self.store_mode = '.pee'
            self.h5 = False
            self.check_pee_red.setEnabled(True)
            
        self.entry_cube_x.setEnabled(self.h5)
        self.entry_cube_y.setEnabled(self.h5)
        self.entry_cube_e.setEnabled(self.h5)
        self.entry_emax.setEnabled(self.h5)
        self.entry_savesteps.setEnabled(self.h5)
        
        self.label_cube_x.setEnabled(self.h5)
        self.label_cube_y.setEnabled(self.h5)
        self.label_cube_e.setEnabled(self.h5)
        self.label_emax.setEnabled(self.h5)
        self.label_savesteps.setEnabled(self.h5)
            
    def _pee(self):
        self.progress.setValue(1)
        self.time_per_event_list = []
        self.time_per_frame_list = []
        
        if self.store_mode == '.h5':
            cube_metadata = {}
            cube_metadata['cube_x'] = int(self.entry_cube_x.text())
            cube_metadata['cube_y'] = int(self.entry_cube_y.text())
            cube_metadata['cube_e'] = int(self.entry_cube_e.text())
            cube_metadata['xmax'] = np.shape(self.MD_mean_frame)[0]
            cube_metadata['ymax'] = np.shape(self.MD_mean_frame)[1]
            cube_metadata['emax'] = int(self.entry_emax.text())
            cube_metadata['savestep'] = int(self.entry_savesteps.text())
        else:
            cube_metadata = None

        hdf5 = False

        mode = self.pee_modes[str(self.comboBox_pee_mode.currentText())]
        data_path = str(self.entry_path_data.text())
        data_path = data_path.replace('\\','/')
        reduced = bool(self.check_pee_red.isChecked())

        path_eval = str(self.entry_path_eval.text())
        path_eval = path_eval.replace('\\','/')
        if not path_eval[-1] == '/':    path_eval += '/'
        eval_name = str(self.entry_eval_name.text())
        print(eval_name)
        if eval_name.split('.')[-1] in ['pee', 'h5', 'hdf5', 'npy', 'npz']:
            save_format = eval_name.split('.')[-1]
            eval_name = eval_name[::-1].split(".", 1)[-1][::-1]
        else:
            if self.store_mode == '.pee':
                save_format = 'pee'
            else:
                save_format = 'npz'
        print(eval_name)
        print(save_format)
        
        sig_fac_1 = float(self.entry_sig_fac_1.text())
        if 'clustering' in mode or mode =='gendreau' or mode=='epic':
            sig_fac_2 = float(self.entry_sig_fac_2.text()) 
        else:                               
            sig_fac_2 = None
            
        if mode == 'gaussian_model_fit' or mode == 'qgmf':    
            squ_a = int(self.entry_squ_a.text())
        else:                               
            squ_a = None
            
        if mode == 'small_unet' or mode == 'big_unet':
            image_scaling = 10/float(self.entry_squ_a.text())                             # 10 is the largest intensity in ADU used in network training
            image_splits = int(self.entry_sig_fac_2.text())
        else:
            image_scaling = 1.
            image_splits = 1
        
        num_rolling_dark = 0
        MD_mode = 'default'
        
        md_frame_choice = str(self.comboBox_pee_dark.currentText())
        if md_frame_choice == 'Master Dark (mean)':
            bg_image = self.MD_mean_frame
            sigma = self.MD_std_frame
        elif md_frame_choice == 'Master Dark (median)':
            bg_image = self.MD_median_frame
            sigma = self.MD_std_frame
        elif md_frame_choice == 'Rolling Dark':
            num_rolling_dark = float(self.entry_rolling_dark_no.text())
            bg_image = None
            sigma = 0
            MD_mode = 'rolling'
        elif md_frame_choice == 'Flat Dark':
            bg_image = float(self.entry_offset.text())
            sigma = float(self.entry_sig.text())
        else:
            bg_image = None
            print('warning: Zero background used for pee')
        
        if self.entry_data_num.text().lower() in ['all', 'inf', '-1', 'oo']: data_num = np.inf
        else:                                   data_num = int(self.entry_data_num.text())
            
        
        self.evaluation.change_settings(data_dir=data_path, eval_dir=path_eval, eval_name=eval_name, pee_mode=mode, MD_mode=MD_mode,
                     md_frame=bg_image, std_frame=sigma, sig_fac_1=sig_fac_1, sig_fac_2=sig_fac_2, squ_a=squ_a, image_scaling=image_scaling,
                     image_splits = image_splits, save_format=save_format, defect_map=self.defect_map, append=False, reduced=reduced,
                     cube_metadata=cube_metadata, rolling_dark_no=num_rolling_dark, num_files_max=data_num, max_wait=15)
        self.evaluation.make_pee()
            
        self.progress.setValue(0)
    

    def _update_pee_stats(self):
        self.progress.setValue(1)
        self.button_save_npx_specs.setEnabled(False)
        stats_mode = str(self.comboBox_pee_stats_mode.currentText())
        event_no = int(float(self.entry_event_no.text()))
        
        eval_path = str(self.entry_path_eval.text())
        if not eval_path[-1] == '/':    eval_path += '/'
        eval_name = str(self.entry_eval_name.text())
        eval_file = eval_path+eval_name
        if eval_file.split('.')[-1] == 'pee':
            do_pee = True
        elif eval_file.split('.')[-1] in ['h5', 'hdf5', 'npy', 'npz']:
            do_pee = False
        else:
            if self.button_store_mode.text() == '.pee':
                do_pee = True
                eval_file += '.pee'
            else:
                do_pee = False
                eval_file += '.h5'
        
        if stats_mode == 'mean event shape':
            E_min = int(self.entry_event_E_min.text())
            E_max = int(self.entry_event_E_max.text())
            out_1,out_2,mode = pt.get_mean_event_shape(eval_file,event_no,E_min,E_max,self.progress)
            if mode == 'gaussian_model_fit' or mode == 'qgmf':
                sigma_xy = out_1
                err_sigma_xy = out_2

                eval_figure = Figure()
                ax1 = eval_figure.add_subplot(111)
                ax1.set_title('Mean Event Shape')
                ax1.set_xlabel('sigma of fit / px')
                ax1.set_ylabel('number of fits')
                bins = np.linspace(0,2,100)
                self.stats = ax1.hist(sigma_xy[0],bins,label = 'x')
                self.stats = ax1.hist(sigma_xy[1],bins,label = 'y')
                ax1.legend(loc='best')
                self.rm_eval()
                self.addmpl_eval(eval_figure)
                
                event_sum = len(sigma_xy[0])
                lines = ['sigma fit ERROR:\t\t{:.0f} events total'.format(event_sum)]
                lines.append('x min.:\t{:.2f}\t\tx max.:\t{:.2f}'.format(min(err_sigma_xy[0]),max(err_sigma_xy[0])))
                lines.append('x mean:\t{:.2f}\t\tx median:\t{:.2f}'.format(np.mean(err_sigma_xy[0]),np.median(err_sigma_xy[0])))
                lines.append('y min.:\t{:.2f}\t\ty max.:\t{:.2f}'.format(min(err_sigma_xy[1]),max(err_sigma_xy[1])))
                lines.append('y mean:\t{:.2f}\t\ty median:\t{:.2f}'.format(np.mean(err_sigma_xy[1]),np.median(err_sigma_xy[1])))
                self.label_event_stats.setText('\n'.join(lines))

            elif mode == 'four_px_area_clustering' or mode == 'four_px_area' or mode == 'clustering' or mode == 'epic' or mode == 'gendreau':
                mes = out_1
                stats = out_2
                
                eval_figure = Figure()
                ax1 = eval_figure.add_subplot(111, projection='3d')
                ax1.set_title('Mean Event Shape')
                ax1.set_xlabel('x axis / px')
                ax1.set_ylabel('y axis / px')
                _xx,_yy = np.meshgrid(np.arange(5),np.arange(5))
                x,y = _xx.ravel(), _yy.ravel()
                width = depth = 1
                self.stats = ax1.bar3d(x, y, np.zeros((5,5)).ravel(), width, depth, mes.ravel(), shade=True)
                self.rm_eval()
                self.addmpl_eval(eval_figure)
                
                event_sum = sum(stats)
                lines = ['size distribution:\t\t{:.0f} events total'.format(event_sum)]
                lines.append('1-px:\t{:.2f}%\t\t2-px:\t{:.2f}%'.format(100.*stats[0]/event_sum,100.*stats[1]/event_sum))
                if len(stats) < 3:
                    stats.append(0)
                if len(stats) < 4:
                    stats.append(0)
                if len(stats) < 5:
                    stats.append(0)   
                lines.append('3-px:\t{:.2f}%\t\t4-px:\t{:.2f}%'.format(100.*stats[2]/event_sum,100.*stats[3]/event_sum))
                lines.append('5-px:\t{:.2f}%\t\t>5-px:\t{:.2f}%'.format(100.*stats[4]/event_sum,100.*sum(stats[5:])/event_sum))
                lines.append('maximum size: {:.0f}-px'.format(len(stats)))
                self.label_event_stats.setText('\n'.join(lines))
            else:
                pass
        
        elif stats_mode == 'event intensity distribution':
            E_min = int(self.entry_event_E_min.text())
            E_max = int(self.entry_event_E_max.text())
            n_min = int(self.entry_event_n_min.text())
            n_max = int(self.entry_event_n_max.text())
            sum_max_list, mode = pt.get_event_intensity_distribution(eval_file,event_no,E_min,E_max,n_min,n_max,self.progress)
            if sum_max_list is not None:
                if len(sum_max_list)>0:
                    if mode == 'gaussian_model_fit' or mode == 'qgmf':
        #                sigma_xy = out_1
        #                err_sigma_xy = out_2
        #
        #                eval_figure = Figure()
        #                ax1 = eval_figure.add_subplot(111)
        #                ax1.set_title('Mean Event Shape')
        #                ax1.set_xlabel('sigma of fit / px')
        #                ax1.set_ylabel('number of fits')
        #                bins = np.linspace(0,2,100)
        #                self.stats = ax1.hist(sigma_xy[0],bins,label = 'x')
        #                self.stats = ax1.hist(sigma_xy[1],bins,label = 'y')
        #                ax1.legend(loc='best')
        #                self.rm_eval()
        #                self.addmpl_eval(eval_figure)
        #                
        #                event_sum = len(sigma_xy[0])
        #                lines = ['sigma fit ERROR:\t\t{:.0f} events total'.format(event_sum)]
        #                lines.append('x min.:\t{:.2f}\t\tx max.:\t{:.2f}'.format(min(err_sigma_xy[0]),max(err_sigma_xy[0])))
        #                lines.append('x mean:\t{:.2f}\t\tx median:\t{:.2f}'.format(np.mean(err_sigma_xy[0]),np.median(err_sigma_xy[0])))
        #                lines.append('y min.:\t{:.2f}\t\ty max.:\t{:.2f}'.format(min(err_sigma_xy[1]),max(err_sigma_xy[1])))
        #                lines.append('y mean:\t{:.2f}\t\ty median:\t{:.2f}'.format(np.mean(err_sigma_xy[1]),np.median(err_sigma_xy[1])))
        #                self.label_event_stats.setText('\n'.join(lines))
                        print('event intensity distribution not supported for gaussian model fit or quick gaussian model fit')
                    
                    elif mode == 'four_px_area_clustering' or mode == 'four_px_area' or mode == 'clustering' or mode == 'epic' or mode == 'gendreau':
                        eval_figure = Figure()
                        ax1 = eval_figure.add_subplot(111)
                        ax1.set_title('Event Shape Distribution')
                        ax1.set_xlabel('Event Seed Intensity / ADU')
                        ax1.set_ylabel('Event Intensity / ADU')
                        #mima_li = lambda l: [int(np.min(l)),int(np.ceil(np.max(l)))]
                        #bins = [np.arange(*mima_li(sum_max_list[:,1])),np.arange(*mima_li(sum_max_list[:,0]))]
                        if E_min == -1: E_min = 0
                        if E_max == -1: E_max = np.max(sum_max_list[:,0])
                        bins = np.linspace(E_min,E_max,1000)
                        h,xedges,yedges,eid = ax1.hist2d(sum_max_list[:,1],sum_max_list[:,0],bins=bins,cmin=1,norm=LogNorm())
                        md_cb_2 = eval_figure.colorbar(eid, ax=ax1)
                        md_cb_2.set_label('log(event number)')
                        
                        self.rm_eval()
                        self.addmpl_eval(eval_figure)
            else:
                pass        
        
        elif stats_mode == 'spectrum':
            self.button_save_npx_specs.setEnabled(True)
            E_min = int(self.entry_event_E_min.text())
            E_max = int(self.entry_event_E_max.text())

            if do_pee:
                out_1, out_2, mode = pt.get_spectrum(eval_file, event_no, Emin_Emax=(E_min,E_max), progress_bar=self.progress)
            else:
                out_1 = pt.get_spectrum_from_cube(eval_file, e_roi=(E_min, E_max))
                out_2 = None
            
            eval_figure = Figure()
            ax1 = eval_figure.add_subplot(111)
            ax1.plot(out_1,color='blue', drawstyle='steps-mid')
            
            self.n_spec_save = {'sum':out_1}
            if out_2 is not None:
                n_min = int(self.entry_event_n_min.text())
                n_max = int(self.entry_event_n_max.text())
                if n_max == -1: n_max = len(out_2)
                if mode != 'gaussian_model_fit' and mode != 'qgmf':
                    for i,histo in enumerate(out_2[n_min-1:n_max]):
                        ax1.fill_between(np.array(range(len(histo))),histo,0, step='mid')
                        ax1.plot(np.array(range(len(histo))),histo,label=str(n_min+i)+'-px: '+str(np.sum(histo)), drawstyle='steps-mid')
                        self.n_spec_save[str(n_min+i)] = histo

            ax1.set_ylim(ymin=1)
            ax1.set_xlabel('photon event intensity / ADU')
            ax1.set_ylabel('number of photon events')
            ax1.set_yscale('log', nonpositive='clip')
            ax1.legend(loc='best')
            self.rm_eval()
            self.addmpl_eval(eval_figure)
        
        elif stats_mode == 'photon flux evolution':
            try:
                SF_list, photon_flux_evolution = pt.get_photon_flux_evolution(eval_file, event_no, progress_bar=self.progress)
            except:
                print("format of pee file too old, information not accessible atm.")
            else:
                if (SF_list is not None) or (photon_flux_evolution is not None):
                    eval_figure = Figure()
                    ax1 = eval_figure.add_subplot(111)
                    ax1.plot(SF_list,photon_flux_evolution)
                    ax1.set_xlabel('frame number')
                    ax1.set_ylabel('number of photon events')
                    self.rm_eval()
                    self.addmpl_eval(eval_figure)
                
        self.progress.setValue(0)      
            
    def _save_npx_specs(self):
        dialog = QtWidgets.QFileDialog(self)
        output = str(dialog.getExistingDirectory(self, 'open file', self.last_path))    ### function for a folder-GUI
        try:    data_path, types = output
        except: data_path = output
        if not data_path == '':
            self.last_path = data_path
        for key in self.n_spec_save:
            np.savetxt(data_path+'/'+key+'.txt',self.n_spec_save[key])
        
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    specFit_GUI = pee_GUI_main()
    specFit_GUI.show()
    app.exec_()

if __name__ == '__main__':
    main()
