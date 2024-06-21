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
import numpy as np
import time
import glob
import os
import copy
from PyQt5 import QtGui as QG, QtCore as QC
from matplotlib import pyplot as plt
from . import pee
from . import gaussian_model_fit as gmf
from . import pee_tools as pt
from . import misc_tools as mt
import h5py
try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print('pytorch not installed. Neural networks not available')

class emitter_object(QC.QObject):
    send_metadata = QC.pyqtSignal(dict)



class pee_reader(object):
    """
    class to read pee file and return pee data for specific images as numpy arrays
    Parameters
    ----------
    file_name : str
    complete file path.  Allowed file extensions are:
    .h5, .hdf5, .npy, .npz. WARNING: npy-format cannot be checked for memory usage
    """
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.metadata = self.get_metadata(file_name)
    
    
    def get_metadata(self, file_name):
        """
        load only metadata of pee or cube file and return as dictionary.

        Parameters
        ----------
        file_name : str
            complete file path.  Allowed file extensions are:
            .h5, .hdf5, .npy, .npz. WARNING: npy-format cannot be checked for memory usage

        Returns
        -------
        metadata : dict
            dictionary with metadata.

        """
        #import psutil
        if file_name.split('.')[-1] in ['h5', 'hdf5']:
            with h5py.File(file_name, 'r') as f:
                try: metadata = dict(f.attrs)
                except: metadata = None
        elif file_name.split('.')[-1] == 'npy':
            try:    [cube, metadata] = np.load(file_name, allow_pickle=True)
            except: 
                cube = np.load(file_name, allow_pickle=True)
                metadata = None
        elif file_name.split('.')[-1] == 'npz':
            data = np.load(file_name, allow_pickle=True)
            metadata = data['attrs'].item()
        elif file_name.split('.')[-1] == 'pee':
            metadata = {'sigma factors':str, 'mean sigma':float, 'mean dark':float, 'dark mode':str, 'squ_a':float, 'mode':str}
            line_no = 0
            with open(file_name, 'r') as pee:
                for line in pee:
                    ### extract meta data
                    if line_no == 0:
                        line = line.split(',')
                        for entry in line[1:]:
                            try:    metadata[entry.split(':')[0].strip()] = metadata[entry.split(':')[0].strip()] = metadata[entry.split(':')[0].strip()](entry.split(':')[1].strip())
                            except: metadata[entry.split(':')[0].strip()] = None
                        try:    metadata['T1'] = float(metadata['sigma factors'].split('-')[0])
                        except: metadata['T1'] = None
                        try:    metadata['T2'] = float(metadata['sigma factors'].split('-')[1])
                        except: metadata['T2'] = None
                        metadata['pee_file'] = os.path.basename(file_name)
                        metadata.pop('sigma factors')
                    elif line_no == 1:
                        line = line.split()
                        if line[0] == 'SI':
                            x_max = int(line[1])
                            y_max = int(line[2])
                            metadata['xmax'] = x_max
                            metadata['ymax'] = y_max
                    else:
                        break
                    line_no += 1
        else:
            raise IOError(f'{file_name.split(".")[-1]} not supported for loading pee cube')
        return metadata
    
    def get_x_y_V(self, imnum:int):
        """
        extract position (x,y) and intensity of events in specified image

        Parameters
        ----------
        imnum : int
            index of image.

        Returns
        -------
        numpy.array(x,y,V).

        """
        x = []
        y = []
        v = []
        extract = False
        with open(self.file_name, 'r') as pee:
            for line in pee:
                line = line.split()
                if line[0] == 'SF':
                    if int(line[1]) == imnum:
                        extract = True
                    else:                       # if extract is true and next SF appears: break
                        if extract:
                            break
                if extract and line[0] == 'HT':
                    x.append(float(line[1]))
                    y.append(float(line[2]))
                    v.append(float(line[3]))
        return np.array([x,y,v])


    
class pee_class(object):
    """
    class for photon event evaluation. Emits stats during evaluation. See
    example.

    Parameters
    ----------
    data_dir : str
        data directory or choice which will be forwarded to glob.glob(). E.g. dir/*.tif.
    eval_dir : str
        folder to store the data.
    eval_name : str
        evaluation name of pee file (without extension).
    pee_mode : str
        possibilities: 'four_px_area', 'clustering', 'four_px_area_clustering', 'qgmf'
        'gaussian_model_fit', 'gendreau', 'epic'. If torch is installed, also 'small_unet'
        and 'big_unet'.
    MD_mode : str
        mode to decide what master dark to use. If 'rolling' the master dark
        frame will be calculated from measurement frames. no md_frame and std_frame
        is necessary. Otherwise, those will be applied.
    md_frame : array, optional
        frame of pixelwise dark current values. The default is None.
    std_frame : array, optional
        frame of pixelwise noise values. The default is None.
    sig_fac_1 : float, optional
        first noise threshold. The default is 6..
    sig_fac_2 : float, optional
        second noise threshold. The default is 3.
    squ_a : int, optional
        edge size in pixel for GMF and to define boarder: box around center hit position
        with edge = squ_a must fit into image. The default is 3.
    image_scaling : float, optional
        scaling factor for image and standard deviation image for neural network evaluation (only!).
        The default is 1.
    image_splits : int, optional
        used for neural network only. Split image into image_splits**2 subimages.
        The default is 1.
    save_format : str, optional
        format of saved data. The default is 'pee'.
    defect_map : array, optional
        if given, regions with red channel are not evaluated. The default is None.
    append : bool, optional
        if True, data is appended to existing pee file. The default is False.
    reduced : bool, optional
        if False, intensities and positions of each pixel are stored in pee file. The default is True.
    cube_metadata : dict, optional
        got values to define cube dimension and conversion ['cube_x, cube_y, cube_e, xmax, ymax, emax']
    rolling_dark_no : int, optional
        number of frames used for rolling dark frame calculation. The default is 20.
    num_files_max : int, optional
        maximum number of frames to evaluate. The default is np.inf.
    max_wait : int, optional
        time to wait for new data in s. The default is 15.

    Example
    -------
    >>>  from pee.base import pee_class
    >>>  def print_signal(status):
    >>>      print(f"timing: {status['evaluation time']:.3f} s")
    >>> data_dir = "datafolder/*.tsv"
    >>> eval_dir = "tmp"
    >>> eval_name = "test"
    >>> pee_mode = "four_px_area"
    >>> MD_mode = "rolling"
    >>>  evaluation = pee_class(data_dir, eval_dir, eval_name, pee_mode, MD_mode, rolling_dark_no=5)
    >>>  evaluation.emitter.send_metadata.connect(print_signal)
    >>>  evaluation.make_pee(10)

    """
    def __init__(self, data_dir, eval_dir, eval_name, pee_mode, MD_mode,
                 md_frame=None, std_frame=None, sig_fac_1=6., sig_fac_2=3., squ_a=3,
                 image_scaling = 1., image_splits = 1, save_format='pee', defect_map=None, append=False, 
                 reduced=True, cube_metadata=None, rolling_dark_no=20, 
                 num_files_max=np.inf, max_wait=15):

        if pee_mode in ['small_unet', 'big_unet'] and not TORCH_AVAILABLE:
            raise ImportError('cannot use network for evaluation. Torch is not available. Choose another pee_mode or '
                              'install pytorch ("https://pytorch.org/get-started/locally/")')
        # necessary settings
        if os.path.isdir(data_dir):
            data_dir = os.path.join(data_dir, '*.*')
        self.data_dir = data_dir
        self.eval_dir = eval_dir
        self.eval_name = eval_name
        self.pee_mode = pee_mode
        self.MD_mode = MD_mode
        # settings with given save default
        self.md_frame = md_frame
        self.std_frame = std_frame
        self.sig_fac_1 = sig_fac_1
        self.sig_fac_2 = sig_fac_2
        self.squ_a = squ_a
        self.image_scaling = image_scaling
        self.image_splits = image_splits
        self.save_format = save_format
        self.defect_map = defect_map
        self.append = append
        self.reduced = reduced
        self.cube_metadata = cube_metadata
        self.cube = None
        self.rolling_dark_no = rolling_dark_no
        self.num_files_max = num_files_max
        self.max_wait = max_wait
        
        # initializing
        self.cube_save_formats = ['h5', 'hdf5', 'npy', 'npz']
        self.num_files_evaluated = 0
        self.rolling_dark_frames = []
        self.emitter = emitter_object()
        if not self.md_frame is None:
            self.mean_dark = np.mean(self.md_frame)
        else:
            self.mean_dark = -1
        if not self.std_frame is None:
            self.mean_sigma = np.mean(self.std_frame)
        else:
            self.mean_sigma = -1
        self.close_images = True
        self.networks = dict(small_unet = 'UNetMod__S__n0-45_Server_23_base',
                             big_unet = 'UNet__S__n0-45_Server_18_base')
        self.networks_tmp = copy.deepcopy(self.networks)
    

    def change_settings(self, **kwargs):
        """
        change pee class settings.
    
        Parameters
        ----------
        data_dir : str
            data directory or choice which will be forwarded to glob.glob(). E.g. dir/*.tif.
        eval_dir : str
            folder to store the data.
        eval_name : str
            evaluation name of pee file (without extension).
        pee_mode : str
            possibilities: 'four_px_area', 'clustering', 'four_px_area_clustering', 'qgmf',
            'gaussian_model_fit', 'gendreau', 'epic'. If torch is installed, also 'small_unet'
            and 'big_unet'.
        MD_mode : str
            mode to decide what master dark to use. If 'rolling' the master dark
            frame will be calculated from measurement frames. no md_frame and std_frame
            is necessary. Otherwise, those will be applied.
        md_frame : array
            frame of pixelwise dark current values. 
        std_frame : array
            frame of pixelwise noise values.
        sig_fac_1 : float
            first noise threshold.
        sig_fac_2 : float
            second noise threshold.
        squ_a : int
            edge size in pixel for GMF and to define boarder: box around center hit position
            with edge = squ_a must fit into image.
        image_scaling : float, optional
            scaling factor for image and standard deviation image for neural network evaluation (only!).
            The default is 1.
        image_splits : int, optional
            used for neural network only. Split image into image_splits**2 subimages.
            The default is 1.
        save_format : str
            format of saved data.
        defect_map : array
            if given, regions with red channel are not evaluated.
        append : bool
            if True, data is appended to existing pee file.
        reduced : bool
            if False, intensities and positions of each pixel are stored in pee file.
        cube_metadata : dict, optional
            got values to define cube dimension and conversion ['cube_x, cube_y, cube_e, xmax, ymax, emax']
        rolling_dark_no : int
            number of frames used for rolling dark frame calculation
        num_files_max : int
            maximum number of frames to evaluate
        max_wait : int
            time to wait for new data in s  
       """
        if 'pee_mode' in kwargs:
            if kwargs['pee_mode'] in ['small_unet', 'big_unet'] and not TORCH_AVAILABLE:
                raise ImportError('cannot use network for evaluation. Torch is not available. Choose another pee_mode or'
                                  'install pytorch ("https://pytorch.org/get-started/locally/")')
        for key in kwargs:
            try:
                self.__dict__.update([(key, kwargs[key])])
            except:
                print('WARNING in pee_class.change_settings:', key, 'cannot be changed!')
                

    def get_files(self):
        file_list = glob.glob(self.data_dir)
        if len(file_list) == 0:
            raise IOError(f'no files in folder {self.data_dir}')
        elif len(file_list) < 10000:
            file_list = list(mt.sort_files(file_list, mode='created')[0])
        else:
            print(f'{len(file_list)} files in folder. Not sorted w.r.t. creation time.')
        return file_list


    def write_header(self, x_max, y_max):
        mt.mkdir(self.eval_dir)
        header = ", ".join([self.eval_name,
                f"sigma factors: {self.sig_fac_1}-{self.sig_fac_2}",
                f"mean sigma: {self.mean_sigma:.3f}",
                f"mean dark: {self.mean_dark:.3f}",
                f"dark mode: {self.MD_mode}",
                f"squ_a: {self.squ_a}",
                f"mode: {self.pee_mode}\n"])
        if self.save_format == 'pee':
            with open(os.path.join(self.eval_dir, self.eval_name + ".pee"), 'w') as pee_file:
                pee_file.write(header)
                pee_file.write("SI"+ " "+ str(x_max)+ " "+  str(y_max)+ "\n")
        elif self.save_format in self.cube_save_formats:
            self.cube_metadata['T1'] = self.sig_fac_1
            self.cube_metadata['T2'] = self.sig_fac_2
            self.cube_metadata['mean sigma'] = self.mean_sigma
            self.cube_metadata['mean dark'] = self.mean_dark
            self.cube_metadata['dark mode'] = self.MD_mode
            self.cube_metadata['squ_a'] = self.squ_a
            self.cube_metadata['mode'] = self.pee_mode
            

    def make_single_pee(self, clean_image, std_frame):
        if self.pee_mode == 'clustering':
            photon_events = pee.clustering(clean_image, self.sig_fac_1, self.sig_fac_2, std_frame, defect_map=self.defect_map)
        elif self.pee_mode == 'four_px_area':
            photon_events = pee.four_px_area(clean_image, float(self.sig_fac_1), std_frame, defect_map=self.defect_map)
        elif self.pee_mode == 'four_px_area_clustering':
            photon_events = pee.four_px_area_clustering(clean_image, self.sig_fac_1, self.sig_fac_2, std_frame, defect_map=self.defect_map)
        elif self.pee_mode == 'gendreau':
            photon_events = pee.asca(clean_image, self.sig_fac_1, self.sig_fac_2, std_frame, defect_map=self.defect_map)
        elif self.pee_mode == 'epic':
            photon_events = pee.epic(clean_image, self.sig_fac_1, self.sig_fac_2, std_frame, defect_map=self.defect_map)
        elif self.pee_mode == 'qgmf':
            photon_events, recon_image, residuum_image, fails = pee.qgmf(clean_image, self.sig_fac_1, self.squ_a, std_frame, defect_map=self.defect_map)    
        elif self.pee_mode == 'gaussian_model_fit':
            bounds = ([0, -np.inf, -self.squ_a, -self.squ_a, -self.squ_a, -self.squ_a],
                      [self.squ_a**2*2**16, np.inf, self.squ_a, self.squ_a, self.squ_a, self.squ_a])
            [photon_events, recon_image, residuum_image, fails] = gmf.get_photon_events(clean_image, self.sig_fac_1, std_frame, self.squ_a,
                                                                                       None, None, bounds, True, sigma=[self.mean_sigma]*self.squ_a**2)
        elif self.pee_mode in self.networks:
            photon_events, recon_image = pee.NN(clean_image, self.sig_fac_1, std_frame, self.networks_tmp[self.pee_mode], image_scaling=self.image_scaling, image_splits=self.image_splits, defect_map=self.defect_map)
        else:
            print(self.pee_mode, self.pee_mode=='small_unet')
            print('pee mode', self.pee_mode,'not supported. Frame skipped.')
            photon_events = []
        return photon_events
    
    
    def get_dark_data(self):
        if self.MD_mode == 'rolling':
            dark_list = [item[1] for item in self.rolling_dark_frames]
            # md_frame = np.median(dark_list, axis=0)
            # std_frame = np.std(dark_list, axis=0)
            std_frame, md_frame = mt.median_standard_deviation(dark_list, axis=0)
            return md_frame, std_frame
        else:
            return self.md_frame, self.std_frame
                    
                    
    def make_pee(self, N=None):
        evaluation_time_s = time.time()
        if self.save_format in self.cube_save_formats:
            assert self.cube_metadata is not None
            self.cube = np.zeros((self.cube_metadata['cube_x'], self.cube_metadata['cube_y'], 
                             self.cube_metadata['cube_e']), dtype=np.uint16)
        else:
            self.cube = None
            
        if N is None:
            N = self.num_files_max
        self.rolling_dark_no = int(self.rolling_dark_no)
        start_pee_time = time.time()
        self.rolling_dark_frames = []
        wait_time = 0
        wait_time_MD = 0
        # file_list = self.get_files()
        file_list_o = []
        file_list_md_o = []
        self.num_files_evaluated = 0
        self.no_of_photon_events = []
        total_photon_events = 0
        status = {}
        status["current frame"] = ''
        status["frame number"] = 0
        status["frames todo"] = 0
        status["photon number"] = 0
        status["evaluation time"] = 0
        status["total photon number"] = 0
        time_per_frame = []
        time_per_event = 0
        
        # for nn only:
        pee_dir = os.path.dirname(pee.__file__)
        if self.pee_mode in self.networks:
            self.networks_tmp[self.pee_mode] = torch.jit.load(os.path.join(pee_dir, 'nn_models', self.networks[self.pee_mode]+'.pt'))
        
        while wait_time < self.max_wait and self.num_files_evaluated < N:
            
            if self.MD_mode == 'rolling':
                ### fill buffer dictionary for master dark creation until it has rolling_dark_no entries
                while wait_time_MD < self.max_wait and len(self.rolling_dark_frames) < self.rolling_dark_no:
                    new_pee_time = time.time()
                    file_list_md = self.get_files()
                    file_list_md_n = [f for f in file_list_md if not f in file_list_md_o]
                    if len(file_list_md_n) > 0:
                        start_pee_time = time.time()
                        print('start loading data for rolling master dark ...')
                    for f in file_list_md[:self.rolling_dark_no]:
                        if f not in [item[0] for item in self.rolling_dark_frames]:
                            self.rolling_dark_frames.append((f, mt.load_image_file(f)))
                    file_list_md_o = file_list_md.copy()
                    wait_time_MD = new_pee_time-start_pee_time
                if len(self.rolling_dark_frames) < self.rolling_dark_no:
                    print('WARNING: Not enough measurement frames to create master dark. Aborting ...')
                    break
            new_pee_time = time.time()
            # get new files only
            file_list = self.get_files()
            file_list_n = [f for f in file_list if not f in file_list_o]
            if not N == np.inf:      
                file_list_n = file_list_n[:N-self.num_files_evaluated]
            time.sleep(0.1)
            if len(file_list_n) > 0:
                start_pee_time = time.time()
                status["frames todo"] += len(file_list_n)
                for f in file_list_n:
                    start_frame_time = time.time()
                    image = mt.load_image_file(f)
                    # write header after 1st file is loaded
                    if self.num_files_evaluated == 0 and not self.append:
                        self.write_header(*np.shape(image))
                    # update rolling dark list
                    if self.MD_mode == 'rolling':
                        if self.num_files_evaluated > self.rolling_dark_no:
                            del self.rolling_dark_frames[0]
                            self.rolling_dark_frames.append((f, image))
                    master_dark, std_frame = self.get_dark_data()
                    clean_image = np.array(image, dtype='float64') - np.array(master_dark, dtype='float64')
                    photon_events = self.make_single_pee(clean_image, std_frame)
                    # save
                    self.save_photon_events(photon_events, self.num_files_evaluated)
                    # stats
                    self.num_files_evaluated += 1
                    photon_num = len(photon_events)
                    self.no_of_photon_events.append(photon_num)
                    total_photon_events += photon_num
                    
                    time_per_frame.append(time.time()-start_frame_time)
                    # if photon_num == 0:
                    #     time_per_event = 0
                    # else:
                    #     time_per_event = time_per_frame[-1]/photon_num
                    
                    status["current frame"] = f
                    status["frame number"] = self.num_files_evaluated
                    status["photon number"] = photon_num
                    status["evaluation time"] = time_per_frame[-1]
                    status["total photon number"] = total_photon_events
                    print(f'file number {self.num_files_evaluated} ({f.split(os.sep)[-1]}) has {photon_num} photon events. Total: {total_photon_events:.4e}')
                    self.send_status(status)
                    
            file_list_o = file_list.copy()            
            wait_time = new_pee_time-start_pee_time
        if self.save_format in self.cube_save_formats:
            self.save_photon_events(photon_events, self.num_files_evaluated, forcesave=True)
        status["total evaluation time"] = time.time()-evaluation_time_s
        self.print_control(status)
        print('evaluation finished')
        self.networks_tmp = copy.deepcopy(self.networks)
        
        
    def save_photon_events(self, photon_events, file_num, forcesave=False):
        if self.save_format == 'pee':
            pt.save_photon_events_to_pee(os.path.join(self.eval_dir, self.eval_name + ".pee"), file_num, photon_events, self.pee_mode, self.reduced)
        elif self.save_format in self.cube_save_formats:
            self.cube = pt.add_photon_events_to_cube(self.cube, photon_events,
                                           self.cube_metadata['xmax'], self.cube_metadata['ymax'], self.cube_metadata['emax'])
            try:    savestep = self.cube_metadata['savestep']
            except: savestep = 1
            if file_num % savestep == 0 or forcesave:
                pt.save_cube(os.path.join(self.eval_dir, self.eval_name + "." + self.save_format), self.cube, append=False, meta_data=self.cube_metadata)
            
    
    def send_status(self, status):
        self.emitter.send_metadata.emit(status)
        
        
    def print_control(self, status):
        mt.mkdir(os.path.join(self.eval_dir, self.eval_name+"_control"))
        plt.figure('number of photon events')
        plt.clf()
        plt.plot(self.no_of_photon_events, 'ro')
        plt.ylabel('detected photons / frame')
        plt.xlabel('file number')
        plt.savefig(os.path.join(self.eval_dir, self.eval_name+"_control", "flux_evolution.png"))
        np.savetxt(os.path.join(self.eval_dir, self.eval_name+"_control", "flux_evolution.dat"), self.no_of_photon_events)
        out_str = ''
        out_str += f'frames evaluated: {status["frame number"]}\n'
        out_str += f'evaluation time: {time.strftime("%H:%M:%S", time.gmtime(status["total evaluation time"]))}\n'
        out_str += f'total photon number: {status["total photon number"]:.0f} ~ {status["total photon number"]:.3e}\n'
        with open(os.path.join(self.eval_dir, self.eval_name+"_control", "report.dat"), 'w') as rep:
            rep.write(out_str)
        if self.close_images:
            plt.close('all')