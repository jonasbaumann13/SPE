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

import numpy as np
import os
import h5py
from numba import jit
from . import misc_tools as mt


def get_spe_key_no(file_path, sample_size=4096, key='\n'):
    with open(file_path, 'r') as spe_file:
        sample = spe_file.read(sample_size)
        lines_in_sample = sample.count(key)
    file_len = int(os.path.getsize(file_path)/sample_size*lines_in_sample)
    return file_len


def _get_mode(eval_file):
    """
    returns the mode string from a spe-file header
    
    Parameters
    ----------
    eval_file : string
        full path to the .spe file
    
    Author
    ------
    Jonas Baumann
    """
    with open(eval_file,'r') as spe_file:
        line = spe_file.readline()
        sl = line.split(', ')
        for entry in sl:
            e = entry.split(': ')
            if e[0] == "mode":
                mode = e[1].replace(',','')
    return mode.lower().strip()


@jit(nopython=True,cache=True)
def contains_box(x,y,x_max,y_max,squ_a):
    """
    checks, if a square box completely fits into a 2d-array

    Parameters
    ----------
    x : int
        x coordinate of the square box
    y : int
        y coordinate of the square box
    x_max : int
        2d-array size in x direction
    y_max : float
        2d-array size in y direction
    squ_a : int
        edge length of square box in pixel sizes, which should fit into the 2d-array

        
    Returns
    -------
    returns True if box fits in, False otherwise

    Author
    ------
    Jonas Baumann
    """
    if (int(squ_a/2) <= x and x < x_max-int(squ_a/2)):
        if(int(squ_a/2) <= y and y < y_max-int(squ_a/2)):                    # define border for evaluation
            return True
    return False


def remove_px_info_from_spe(eval_file_name, copy_file_name):
    """
    removes all lines with "PX" from the spe file, saving disc space
    
    Parameters
    ----------
    eval_file_name : string
        complete file path for the input file
    copy_file_name : string
        complete file path for the output file

    Author
    ------
    Jonas Baumann
    """
    in_file = open(eval_file_name, 'r')
    out_file = open(copy_file_name, 'w')
    
    for line in in_file:
        if not line.split()[0] == 'PX':
            out_file.write(line)
    
    in_file.close()
    out_file.close()


def combine_spe(spe_1, spe_2, spe_out):
    """
    combine two spe files. The header of second spe file is discarded. Frame numbers
    'SF' of spe_2 are corrected by final frame number of spe_1.

    Parameters
    ----------
    spe_1 : str
        path to first spe file (header of this is used for new file).
    spe_2 : str
        path to second spe file.
    spe_out : str
        path for output spe file.

    Returns
    -------
    None.

    """
    SF = 0
    start_p2 = False
    with open(spe_out, 'w') as po:
        with open(spe_1, 'r') as p1:
            for line in p1:
                po.write(line)
                if line.split()[0] == 'SF': SF = int(line.split()[1])
        with open(spe_2, 'r') as p2:
            for line in p2:
                line_s = line.split()
                if line_s[0] == 'SF': start_p2 = True
                if start_p2:
                    if line_s[0] == 'SF':   line_s[1] = '{}'.format(int(line_s[1])+SF+1)
                    po.write(' '.join(line_s)+'\n')
                    
                    
def save_photon_events_to_spe(eval_file_name, file_nr, spe_data, spe_mode, reduced=False):
    """
    save the data array from photon event evaluation (spe) to file as list
    
    Parameters
    ----------
    eval_file_name : string
        complete file path for the output file
    file_nr : float or string
        data seperator e.g. to distinguis events from different frames of a ccd
    spe_data : list
        list returned from a spe algorithm
    spe_mode : string
        method that was used. valid strings are "clustering", "four_px_area",
        "four_px_area_clustering", "qgmf", "gaussian_model_fit"
    reduced : boolean
        if True, only total event information is saved after keyword "HT", data of
        single pixels after "PX" is not stored.
        
    File Format
    -----------
    lines beginning with "SF" indicate a new data set (see file_nr). lines beginning
    with "HT" indicate an detected event
    
    In case of spe_mode = "clustering", "four_px_area", "four_px_area_clustering"
    the format is as follows: x-position, y-position, total intensity, number of
    contributing pixels
    
    Additionally, the position and relative intensity of each pixel of every event
    is saved in subsequent lines beginning with "PX"
    
    In case of spe_mode = qgmf or gaussian_model_fit, the format is: x-position, y-position, 
    total intensity, sigma width in x-direction, sigma width in y-direction
    
    Each HT-line follows a line beginning with "ER" with the estimated standard
    deviation of each parameter from the fit.

    Author
    ------
    Jonas Baumann
    """
    of = open(eval_file_name, 'a')
    of.write("SF"+ " "+ str(file_nr)+ " "+ str(len(spe_data))+ "\n")
    
    if spe_mode in ['clustering', 'four_px_area', 'four_px_area_clustering', 'gendreau', 'epic']:
        for x,y,summe,n,px_int,px_pos in [[a,b,c,d,e,f] for [a,b,c,d,e,f] in spe_data]:
                of.write("HT"+ " "+ str(x)+ " "+  str(y)+ " " + str(summe) + " " + str(n) + "\n")
                if not reduced:
                    for no in range(len(px_int)):
                        of.write("PX"+ " " + str(px_pos[no][0]-x)+" "+str(px_pos[no][1]-y)+" "+str(px_int[no]/float(summe))+"\n")
                        
    elif spe_mode in ['qgmf', 'gaussian_model_fit']:
        for x,y,summe,sig_x,sig_y,err,mse in [[a,b,c,d,e,f,g] for [a,b,c,d,e,f,g] in spe_data]:                         # mse is mean squared error sum((a-b)**2)/N
            of.write("HT"+ " "+ str(x)+ " "+  str(y)+ " " + str(summe) + " " + str(sig_x) + " " + str(sig_y) + "\n")
            str_err = " ".join([str(e) for e in err])
            of.write("ER" + " " + str_err + " " + str(mse) + "\n")
            
    elif spe_mode in ['small_unet', 'big_unet']:
        for x,y,summe in [[a,b,c] for [a,b,c] in spe_data]:
                of.write("HT"+ " "+ str(x)+ " "+  str(y)+ " " + str(summe) + "\n")
            
    of.close()


def spefile_to_cube(f, len_x, len_y, len_E, E_max, npx_bounds=None, max_images=np.inf):
    """
    convert spe file to data cube

    Parameters
    ----------
    f : str
        path to spe file.
    len_x : int
        x entries in data cube (aka pixels).
    len_y : int
        y entries in data cube (aka pixels).
    len_E : int
        energy bins in data cube.
    x_max : int
        shape of camera (2048 for CMOS).
    y_max : int
        shape of camera (2048 for CMOS).
    E_max : int
        maximum energy in ADU from spectrum. SHOULD BE MULTIPLE OF len_E!.
    npx_bounds : tuple
        lower and upper limit of pixels contributing to the event
    max_images : TYPE
        number of images to convert, for testing e.g. take not all.

    Returns
    -------
    cube : numpy.ndarray
        data cube with axis (x, y, e).

    """
    head_dic = {'sigma factors':str, 'mean sigma':float, 'mean dark':float, 'dark mode':str, 'squ_a':float, 'mode':str}
    no_images = 0
    line_no = 0
    if npx_bounds is None:
        n1, n2 = 0, np.inf
    else:
        n1, n2 = npx_bounds
        if n2 == -1: n2 = np.inf
    cube = None
    with open(f, 'r') as spe:
        for line in spe:
            ### extract meta data
            if line_no == 0:
                line = line.split(',')
                for entry in line[1:]:
                    try:    head_dic[entry.split(':')[0].strip()] = head_dic[entry.split(':')[0].strip()] = head_dic[entry.split(':')[0].strip()](entry.split(':')[1].strip())
                    except: head_dic[entry.split(':')[0].strip()] = None
                try:    head_dic['T1'] = float(head_dic['sigma factors'].split('-')[0])
                except: head_dic['T1'] = None
                try:    head_dic['T2'] = float(head_dic['sigma factors'].split('-')[1])
                except: head_dic['T2'] = None
                head_dic['npx'] = npx_bounds
                head_dic['spe_file'] = os.path.basename(f)
                head_dic.pop('sigma factors')
                line_no += 1
            
            else:
                line = line.split()
                if len(line) > 0:
                    ### build cube from image size info ('SI')
                    if line[0] == 'SI':
                        x_max = int(line[1])
                        y_max = int(line[2])
                        cube = np.zeros((len_x, len_y, len_E), dtype=np.uint16)
                    ### add data to cube
                    elif cube is not None:
                        if no_images > max_images:
                            break
                        if line[0] == 'SF':
                            no_images += 1
                        if line[0] == 'HT':
                            if float(line[3]) > 0:
                                n = float(line[4])
                                if n1 <= n and n <= n2:
                                    x_ind = int(float(line[1])/x_max*len_x)
                                    y_ind = int(float(line[2])/y_max*len_y)
                                    E_ind = int(float(line[3])/E_max*len_E)
                                    try:    cube[x_ind, y_ind, E_ind] += 1
                                    except: pass
    return cube, head_dic


def add_photon_events_to_cube(cube, spe_data, x_max, y_max, e_max):
    [len_x, len_y, len_e] = np.shape(cube)
    for pd in spe_data:
        if pd[2] > 0:
            x_ind = int(pd[0]/x_max*len_x)
            y_ind = int(pd[1]/y_max*len_y)
            e_ind = int(pd[2]/e_max*len_e)
            try:    cube[x_ind, y_ind, e_ind] += 1
            except: pass
    return cube
            
            
def save_cube(file_name, cube, append=False, meta_data=None):
    import h5py
    """
    save the spe cube to an h5 or numpy file
    data cube of 3 dimensions (x, y, e).
    
    Parameters
    ----------
    file_name : string
        complete file path for the output file. Allowed file extensions are:
        .h5, .hdf5, .npy, .npz. npy-format is not compressed, the others are.
    cube : numpy.ndarray
        data cube with axis (x, y, e)
    
    Author
    ------
    Jonas Baumann
    """
    if append:
        old_cube, old_metadata = load_cube(file_name)
        assert (np.shape(cube) == np.shape(old_cube))
        assert cube.dtype == old_cube.dtype
        cube += old_cube
        if old_metadata is not None:
            try:    old_metadata.update(meta_data)
            except: meta_data = old_metadata
        
    if file_name.split('.')[-1] in ['h5', 'hdf5']:
        with h5py.File(file_name, 'w') as h5:
            h5.create_dataset('cube', data=cube, compression="gzip")
            if meta_data is not None:
                for key in meta_data:
                    try:    h5.attrs[key] = meta_data[key]
                    except: pass
        
    elif file_name.split('.')[-1] == 'npy':
        np.load(file_name, allow_pickle=True)
        np.save(file_name, np.array([cube, meta_data], dtype=object))

    elif file_name.split('.')[-1] == 'npz':
        if meta_data is None:
            meta_data = {}
        meta_data['memory'] = cube.size*cube.itemsize
        np.savez_compressed(file_name, cube=cube, attrs=meta_data)
        
    else:
        raise IOError(f'{file_name.split(".")[-1]} not supported for saving spe cube')
        

def load_cube(file_name):
    """
    read data cube from h5 file

    Parameters
    ----------
    file_name : str
        complete file path.  Allowed file extensions are:
        .h5, .hdf5, .npy, .npz. WARNING: npy-format cannot be checked for memory usage

    Returns
    -------
    cube : numpy.ndarray
        data cube of 3 dimensions (x, y, e).

    """
    import psutil
    if file_name.split('.')[-1] in ['h5', 'hdf5']:
        with h5py.File(file_name, 'r') as f:
            mem_usage = f['cube'].size*np.dtype(f['cube'].dtype).itemsize
            mem_free = psutil.virtual_memory().available
            if mem_usage*1.1 < mem_free:
                cube = np.array(f['cube'])
            else:
                raise MemoryError(f'not enough memory to load the cube. {mem_usage/1e9:.3f} GB neccessary, but only {mem_free/1e9:.3f} GB free.')
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
        mem_free = psutil.virtual_memory().available
        mem_usage = metadata['memory']
        if mem_usage*1.1 < mem_free:
            cube = data['cube']
        else:
            raise MemoryError(f'not enough memory to load the cube. {mem_usage/1e9:.3f} GB neccessary, but only {mem_free/1e9:.3f} GB free.')
    else:
        raise IOError(f'{file_name.split(".")[-1]} not supported for loading spe cube')
    
    return cube, metadata
        
                        
                        
def save_photon_events_to_hdf5(eval_file_name, spe_data, frame_shape, spec_length, split_cubes=True):
    import h5py
    """
    save the data array from photon event evaluation (spe) to a hdf5
    data cube of 3 dimensions (x,y,spectrum).
    
    Parameters
    ----------
    eval_file_name : string
        complete file path for the output file
    spe_data : list
        list returned from an spe algorithm
    frame_shape : tuple of ints
        (x,y) with x and y the maximum dimension of the image used as raw data.
    spec_length : int
        length of the spectra stored in the cube
    split_cubes : boolean, optional
        if True, the data set is split into 4. See Notes for details.
    
    Notes
    -----
    if split_cubes = True, the data cube is splitted into 4 subcubes
    
    Author
    ------
    Jonas Baumann
    """
    if split_cubes: 
        cubes = ['Cube1','Cube2','Cube3','Cube4']
        frame_shape = (int(frame_shape[0]/2.),int(frame_shape[1]/2.),spec_length)
    else:   
        cubes = ['Cube']
        frame_shape = frame_shape + (spec_length,)
    
    if eval_file_name.split('.')[-1] == 'h5':
        with h5py.File(eval_file_name,'w') as h5file:
            for cube in cubes:
                h5file.create_dataset(cube,frame_shape,dtype='uint32',compression="gzip")
            
            if split_cubes:
                for x,y,summe in [[a,b,c] for [a,b,c,d,e,f] in spe_data]:
                    if summe < spec_length:
                        if      x%2 == 0 and y%2 == 0: cube = cubes[0]
                        elif    x%2 == 1 and y%2 == 0: cube = cubes[1]
                        elif    x%2 == 0 and y%2 == 1: cube = cubes[2]
                        elif    x%2 == 1 and y%2 == 1: cube = cubes[3]
                        h5file[cube][int(x/2),int(y/2),int(summe)] += 1               
            else:
                for x,y,summe in [[a,b,c] for [a,b,c,d,e,f] in spe_data]:
                    if summe < spec_length:
                        h5file[cube][int(x),int(y),int(summe)] += 1


def add_photon_events_to_hdf5(eval_file_name, spe_data):
    """
    add the data array from photon event evaluation (spe) to a hdf5
    data cube of 3 dimensions (x,y,spectrum).
    
    Parameters
    ----------
    eval_file_name : string
        complete file path for the output file
    spe_data : list
        list returned from an spe algorithm

    Notes
    -----
    The original data size of the cube (x,y,spectrum) is not changed. This might
    result in loss of data or exceptions if the data is not compatible
    
    Author
    ------
    Jonas Baumann
    """
    import h5py
    if eval_file_name.split('.')[-1] == 'h5':
        with h5py.File(eval_file_name,'a') as h5file:           
            if len(h5file) == 4:
                for x,y,summe in [[a,b,c] for [a,b,c,d,e,f] in spe_data]:
                    if summe < h5file[h5file.keys()[0]].shape[2]:
                        if      x%2 == 0 and y%2 == 0: cube = h5file.keys()[0]
                        elif    x%2 == 1 and y%2 == 0: cube = h5file.keys()[1]
                        elif    x%2 == 0 and y%2 == 1: cube = h5file.keys()[2]
                        elif    x%2 == 1 and y%2 == 1: cube = h5file.keys()[3]
                        h5file[cube][int(x/2),int(y/2),int(summe)] += 1               
            else:
                for x,y,summe in [[a,b,c] for [a,b,c,d,e,f] in spe_data]:
                    if summe < h5file[h5file.keys()[0]].shape[2]:
                        h5file[h5file.keys()[0]][int(x),int(y),int(summe)] += 1


def estimate_key_no_in_file(file_path, sample_size=4096, key='\n'):
    import os
    assert file_path.split('.')[-1] == 'spe'
    with open(file_path, 'r') as spe_file:
        sample = spe_file.read(sample_size)
        lines_in_sample = sample.count(key)
    key_no = int(os.path.getsize(file_path)/sample_size*lines_in_sample)
    return key_no


def make_angle_spectra(spe_data, eval_dir, xrf_lib_ccd, angular_resolution, max_ADU=None, energy_calibration=None, max_counts=1e10, max_frame_no=np.inf, progress_bar=None, chirp=False):
    """
    add the data array from photon event evaluation (spe) to a hdf5
    data cube of 3 dimensions (x,y,spectrum).
    
    Parameters
    ----------
    spe_data : string or numpy.ndarray
        complete file path for the file of type ".spe" or ".h5" or data cube directly
    eval_dir : string
        complete file path to the evaluation directory, where the results
        are saved in
    xrf_lib_ccd : object of axp_tools.xrf_library.detectors.ccd
        detector with all the geometry and bad-pixel-map etc defined.
    angular_resolution : float
        angle in deg defining the angular area used for each spectrum
    max_ADU : int
        number of channels for the spectrum
    energy_calibration : function, optional
        used to convert ADU to keV for each photon. If given, the pixelwise
        efficiency of xrf_lib_ccd will be calculated and applied to every
        detected photon.... note that pileup events are thus given a wrong
        efficiency
    max_counts : float or int, optional
        maximum number of counts per channel, since vanishing efficiencies can
        lead to infinite count numbers
    max_frame_no : int, optional
        maximum number of frames to evaluate, set np.inf to evaluate all frames
        in spe_data
    progress_bar : QtWidgets.QProgressBar, optional
        if set AND chirp=True will show progress
    chirp : boolean, optional
        gives some additional output about progress, rejected photons and plots
        the spectra if set True
   
    Author
    ------
    Jonas Baumann
    """
    import os
    from matplotlib import pyplot as plt
    
    SF = 0
    if chirp and max_frame_no == np.inf:
        max_frame_no = estimate_key_no_in_file(spe_data,sample_size=5*4096,key='SF')
        
    photon_map = np.zeros((xrf_lib_ccd.n1,xrf_lib_ccd.n2))
    xrf_lib_ccd.set_resolution(angular_resolution)
    reject = 0
    
    if isinstance(spe_data, str):
        #### algorithm if data is stored in spe file ####
        if spe_data.split('.')[-1] == 'spe':
            angle_spectra = [[0.]*max_ADU for phi in range(len(xrf_lib_ccd.angles))]
            angle_spectra_raw = [[0.]*max_ADU for phi in range(len(xrf_lib_ccd.angles))]
            spef = open(spe_data)
            for line in spef:
                if SF > max_frame_no:
                    break
                else:
                    sl = line.split()
                    if sl[0] == "HT":
                        x = int(float(sl[1]))           # justified since all values between e.g. 1.0 and 1.9999... are put to bin 1
                        y = int(float(sl[2]))
                        I = int(float(sl[3]))
                        
                        photon_angle = xrf_lib_ccd.angles_2D[y][x]
                        
                        if I < max_ADU:
                            photon_map[int(y)][int(x)] += 1.
                            
                            for i,phi in enumerate(xrf_lib_ccd.angles):
                                if (phi-0.5*xrf_lib_ccd.resolution <= photon_angle) and (photon_angle <= phi+0.5*xrf_lib_ccd.resolution):
                                    # if energy_calibration is given, normalize to detector efficiency (currently e.g. chip absorption, air and Be transmission')
                                    # in any case, also save the uncorrected data for correct sampling uncertainties later in GEXRF profile
                                    if energy_calibration:
                                        eff = xrf_lib_ccd.get_pixel_efficiency(y,x,energy_calibration(I))
                                        if eff > 1e-10:
                                            if 1./eff <= max_counts:
                                                angle_spectra[i][I] += 1./eff
                                            else:
                                                reject += 1
                                        else:
                                            reject += 1
                                    angle_spectra_raw[i][I] += 1.
        
                    if sl[0] == "SF":
                        if chirp:
                            if progress_bar:    progress_bar.setValue(int(100.*SF/max_frame_no))
                            else:               print("{:.1f}%".format(int(100.*SF/max_frame_no)))
                        if not sl[1] == 'END':  SF += 1
        
        #### algorithm if data is stored in h5 file
        elif spe_data.split('.')[-1] == 'h5':
            spe_data, metadata = load_cube(spe_data)
            
            
    if isinstance(spe_data, np.ndarray):
        #### algorithm if data is stored in cube ndarray ####
        angle_spectra = np.zeros(spe_data.shape[2], dtype=np.float64)
        angle_spectra_raw = np.zeros(spe_data.shape[2], dtype=np.float64)
        for i, phi in enumerate(xrf_lib_ccd.angles):
            angle_spectra_raw[i] = np.sum(spe_data[(phi-0.5*xrf_lib_ccd.resolution <= xrf_lib_ccd.angles_2D) * 
                                   (xrf_lib_ccd.angles_2D < phi+0.5*xrf_lib_ccd.resolution)], axis=(1,2))
            if chirp:
                if progress_bar:    progress_bar.setValue(int(100.*i/len(xrf_lib_ccd.angles)))
                else:               print("{:.1f}%".format(int(100.*i/len(xrf_lib_ccd.angles))))
                
        if energy_calibration:
            for I in range(np.shape(spe_data)[2]):
                eff_map = xrf_lib_ccd.calc_pixel_efficiency_map(energy_calibration(I))
                spe_data[:,:,I] /= eff_map
            for i, phi in enumerate(xrf_lib_ccd.angles):
                angle_spectra[i] = np.sum(spe_data[(phi-0.5*xrf_lib_ccd.resolution <= xrf_lib_ccd.angles_2D) * 
                                       (xrf_lib_ccd.angles_2D < phi+0.5*xrf_lib_ccd.resolution)], axis=(1,2))
                if chirp:
                    if progress_bar:    progress_bar.setValue(int(100.*i/len(xrf_lib_ccd.angles)))
                    else:               print("{:.1f}%".format(int(100.*i/len(xrf_lib_ccd.angles))))
    
    
    if chirp:   print(reject,'events rejected')
    os.system("mkdir -p "+eval_dir)
    
    with open(eval_dir+'angle_spectra.txt',"w") as fout:
        fout.write('')
    with open(eval_dir+'angle_spectra.txt',"a") as fout:
        if energy_calibration:
            as_copy = np.copy(angle_spectra)
        else:
            as_copy = np.copy(angle_spectra_raw)
        for j,angle_spec in enumerate(as_copy):
#            if chirp:
#                if progress_bar:    progress_bar.setValue(100.*j/len(angle_spectra))
#                else:   print("{:.1f}%".format(100.*j/len(angle_spectra)))
            np.savetxt(fout,[[xrf_lib_ccd.angles[j]]+list(angle_spec)])
            if chirp:
                plt.figure(0)
                plt.clf()
                x_axis = range(len(angle_spec))
                x_label = 'energy / ADU'
                y_label = 'detected photons'
                if energy_calibration:  
                    x_axis = list(map(energy_calibration,x_axis))
                    x_label = 'energy / keV'
                    y_label = 'detected photons (efficiency corrected)'
                plt.plot(x_axis,angle_spec)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                try:    
                    plt.ylim(bottom=1)
                    plt.yscale('log')
                except: pass
                plt.ylim(bottom=1)
                plt.savefig(eval_dir+'spec_{:.4f}deg.png'.format(xrf_lib_ccd.angles[j]))
        ### add raw spectrum at end of angle_specs if energy_calibration is given, 
        ### for statistical uncertainties later in GEXRF profile
        if energy_calibration is not None:
            for j,angle_spec in enumerate(angle_spectra_raw):
                np.savetxt(fout,[[-xrf_lib_ccd.angles[j]]+list(angle_spec)])
    
    if energy_calibration:
        plt.figure()
        plt.plot(np.sum(angle_spectra,axis=0))
        plt.yscale('log')
        plt.ylim(bottom=1)
        plt.xlim((0,max_ADU))
        plt.savefig(eval_dir+'sum_spec.png')
        np.savetxt(eval_dir+'sum_spec.dat',np.sum(angle_spectra,axis=0))
        plt.figure()
    plt.figure()
    plt.plot(np.sum(angle_spectra_raw,axis=0))
    plt.yscale('log')
    plt.ylim(bottom=1)
    plt.xlim((0,max_ADU))
    plt.savefig(eval_dir+'sum_spec_raw.png')
    np.savetxt(eval_dir+'sum_spec_raw.dat',np.sum(angle_spectra,axis=0))
    
    xrf_lib_ccd.put_photonmap(photon_map)
    xrf_lib_ccd.calc_GEXRF_profile()
    np.savetxt(eval_dir+'angles_solidangles.txt',[xrf_lib_ccd.angles,xrf_lib_ccd.solid_angles])
    xrf_lib_ccd.plot_image(save_path=eval_dir+'photon_map.png')
    


def get_spectrum_from_cube(cube, e_roi=None, xy_roi=None):
    """
    get spectrum of data cube for specific energy and position roi

    Parameters
    ----------
    cube : numpy.ndarray
        data cube of 3 dimensions (x,y,spectrum).
    e_roi : tuple, optional
        tuple of (Emin, Emax) for energy roi. The default is None.
    xy_roi : tuple, optional
        tuple of ((xmin, xmax), (ymin, ymax) for position roi. The default is None.

    Returns
    -------
    sum_spec : numpy.ndarray
        spectrum data.

    """
    if isinstance(cube, str):
        cube, metadata = load_cube(cube)
    if e_roi is None:
        Emin, Emax = 0, np.shape(cube)[2]
    else:
        Emin, Emax = e_roi
        if Emax == -1: Emax = np.shape(cube)[2]
    if xy_roi is None:
        xmin, xmax = 0, np.shape(cube)[0] 
        ymin, ymax = 0, np.shape(cube)[1]
    else:
        if xy_roi[0][0] == -1: xy_roi[0][0] = np.shape(cube)[0]
        if xy_roi[0][1] == -1: xy_roi[0][1] = np.shape(cube)[0]
        if xy_roi[1][0] == -1: xy_roi[1][0] = np.shape(cube)[1]
        if xy_roi[1][1] == -1: xy_roi[1][1] = np.shape(cube)[1]
        xmin = np.min(xy_roi[0])
        xmax = np.max(xy_roi[0])
        ymin = np.min(xy_roi[1])
        ymax = np.max(xy_roi[1])
        
    sum_spec = np.sum(cube[xmin:xmax, ymin:ymax, Emin:Emax], axis=(0,1))
    return sum_spec


def get_pm_from_cube(cube, e_rois=None):
    """
    get photon map of data cube for specific energy  roi

    Parameters
    ----------
    cube : numpy.ndarray
        data cube of 3 dimensions (x,y,spectrum).
    e_rois : dictionary, optional
        tuple of (Emin, Emax) for energy roi. The default is None.

    Returns
    -------
    photon_maps : numpy.ndarray
        photon maps for all e_rois.

    """
    photon_maps = {}
    if e_rois is None:
        e_rois = {'all': None} # takes all energies
    for label in e_rois:
        if e_rois[label] is None:
            Emin, Emax = 0, np.shape(cube)[2]
        else:
            Emin, Emax = e_rois[label]
            if Emax == -1: Emax = np.shape(cube)[2]
            
        photon_maps[label] = np.sum(cube[:, :, Emin:Emax], axis=2, dtype=np.float64)
        
    return photon_maps


def get_spectrum(eval_file, event_no, Emin_Emax=None, xy_roi=None, progress_bar=[], chisq_lim=np.inf, bins=None):
    """
    get spectrum from a spe file.

    Parameters
    ----------
    eval_file : str
        path to file.
    event_no : int
        number of events to evaluate.
    Emin_Emax : tuple, optional
        (minimum energy, maximum energy) for filtering photons. The default is None.
    xy_roi : list of lists, optional
        2x2 matrix for x y position filtering, ((xmin, xmax),(ymin, ymax)). The default is None.
    progress_bar : QtWidgets.QProgressBar, optional
        place to plot the processing progress. The default is [].
    chisq_lim : float, optional
        only used for GMF to reject bad fits. The default is np.inf.

    Returns
    -------
    array
        sum spectrum.
    list
        n-px spectra (so its a list of lists).
    string
        mode of spe evaluation.

    """
    if event_no == -1:
        event_no = get_spe_key_no(eval_file, key="HT")
    n = 0
    if not Emin_Emax is None:
        E_min, E_max = Emin_Emax
    else:
        E_min, E_max = 0, -1
    if E_max == -1:
        E_max = np.inf
    if not xy_roi is None:
        xy_roi = np.array(xy_roi, dtype='float64')
        xy_roi[xy_roi==-1] = np.inf
        x_min = np.min(xy_roi[0])
        x_max = np.max(xy_roi[0])
        y_min = np.min(xy_roi[1])
        y_max = np.max(xy_roi[1])
    else:
        x_min = 0
        x_max = np.inf
        y_min = 0
        y_max = np.inf
    # if x_max == -1: x_max = np.inf
    # if y_max == -1: y_max = np.inf
    mode = _get_mode(eval_file)
    #print(mode)
    
    if mode in ['gaussian_model_fit', 'qgmf', 'small_unet', 'big_unet']:
        last_E = -1
        last_x = -1
        last_y = -1
        pix_events = []
        with open(eval_file, 'r') as spe_file:
            for i,line in enumerate(spe_file):
                if n >= event_no:    break
                sl = line.split()
                if len(sl) > 0:
                    if sl[0] == "HT":
                        last_E = float(sl[3])
                        last_x = float(sl[1])
                        last_y = float(sl[2])
                        if E_min <= last_E and last_E <= E_max:
                            if x_min <= last_x and last_x <= x_max:
                                if y_min <= last_y and last_y <= y_max:
                                    n += 1
                                    pix_events.append(last_E)
                    if sl[0] == "ER":
                        if E_min <= last_E and last_E <= E_max:
                            if x_min <= last_x and last_x <= x_max:
                                if y_min <= last_y and last_y <= y_max:
                                    if float(sl[6])>chisq_lim:
                                        pix_events.pop()
                                        n -= 1
                    if (n%(int(event_no/100))) == 0:
                        if progress_bar:
                            try:    progress_bar.setValue(int(100.*n/event_no))   # if it is progress_bar
                            except: pass
                            try:    progress_bar.emit(int(100.*n/event_no))       # if it is a signal
                            except: pass
                        
        if bins is None:
            hist_max = min([np.percentile(pix_events, 99.5), 1e5])
            bins = np.arange(0,hist_max,1)
        sum_spec, binedge = np.histogram(pix_events,bins=bins)
        return sum_spec, None, mode
    
    elif mode.lower() in ['four_px_area_clustering', 'four_px_area', 'clustering', 'gendreau', 'epic', 'cnn', 'epic_old']:
        n_pix_events = [[] for i in range(1)]
        with open(eval_file,'r') as spe_file:
            for i,line in enumerate(spe_file):
                if n >= event_no:    break
                sl = line.split()
                if len(sl)>0:
                    if sl[0] == "HT":
                        if E_min <= float(sl[3]) and float(sl[3]) <= E_max:
                            if x_min <= float(sl[1]) and float(sl[1]) <= x_max:
                                if y_min <= float(sl[2]) and float(sl[2]) <= y_max:
                                    n += 1
                                    intensity = float(sl[3])
                                    n_px = int(float(sl[4]))
                                    while len(n_pix_events) < n_px:    n_pix_events.append([])
                                    n_pix_events[n_px-1].append(intensity)
                        if (n%(int(event_no/100))) == 0:
                            if progress_bar:    
                                try:    progress_bar.setValue(int(100.*n/event_no))   # if it is progress_bar
                                except: pass
                                try:    progress_bar.emit(100.*n/event_no)       # if it is a signal
                                except: pass
        histo_max = 0
        tmp_max = 0
        for future_histo in n_pix_events:
            if future_histo: tmp_max = min([np.percentile(future_histo, 99.5), 1e5])
            if tmp_max > histo_max: histo_max = tmp_max

        if bins is None:
            bins = np.arange(0,histo_max,1)
        for i in range(len(n_pix_events)):
            n_pix_events[i],bin_edges = np.histogram(n_pix_events[i],bins=bins)
        
        sum_spec = np.zeros(len(bin_edges)-1)
        for histo in n_pix_events:
            sum_spec += histo
        return sum_spec, n_pix_events, mode
    else:
        return None, None, None


def get_mean_event_shape(eval_file, event_no, E_min, E_max, progress_bar):
    if eval_file.split('.')[-1] != 'spe':
        print(f'WARNING: {eval_file.split(".")[-1]} file does not support mean event shape calculation.')
        return None, None, None
    else:
        if E_max == -1:  E_max = np.inf
        if event_no == -1:
            event_no = get_spe_key_no(eval_file, key="HT")
        n = 0
        mode = _get_mode(eval_file)

        if mode == 'gaussian_model_fit':
            sigma_x = []
            sigma_y = []
            err_sigma_x = []
            err_sigma_y = []
            with open(eval_file, 'r') as spe_file:
                for line in spe_file:
                    if n >= event_no:    break
                    sl = line.split()
                    if sl[0] == "HT":
                        if E_min < float(sl[3]) and float(sl[3]) < E_max:
                            n += 1
                            sigma_x.append(float(sl[4]))
                            sigma_y.append(float(sl[5]))
                            progress_bar.setValue(100. * n / event_no)
                            in_range = True
                        else:
                            in_range = False
                    if sl[0] == "ER" and in_range:
                        err_sigma_x.append(float(sl[4]))
                        err_sigma_y.append(float(sl[5]))
            return [sigma_x, sigma_y], [err_sigma_x, err_sigma_y], mode

        elif mode == 'four_px_area_clustering' or mode == 'four_px_area' or mode == 'clustering' or mode == 'gendreau' or mode == 'epic' or mode == 'cnn':
            size = [0]
            mes = np.zeros((5, 5))
            in_range = False
            with open(eval_file, 'r') as spe_file:
                for line in spe_file:
                    if n >= event_no:    break
                    sl = line.split()
                    if sl[0] == "HT":
                        if E_min < float(sl[3]) and float(sl[3]) < E_max:
                            n += 1
                            s = int(sl[4])
                            if len(size) < s:
                                size.append(1)
                            else:
                                size[s - 1] += 1
                            in_range = True
                            progress_bar.setValue(100. * n / event_no)
                        else:
                            in_range = False
                    if sl[0] == "PX" and in_range:
                        try:
                            mes[int(round(float(sl[1])) + 2), int(round(float(sl[2])) + 2)] += float(sl[3])
                        except:
                            pass
            mes /= n
            return mes, size, mode

        else:
            return None, None, None


def get_event_intensity_distribution(eval_file, event_no, E_min, E_max, n_min, n_max, progress_bar=[]):
    mode = _get_mode(eval_file)
    if eval_file.split('.')[-1] != 'spe':
        print(f'WARNING: {eval_file.split(".")[-1]} file does not support event intensity distribution calculation.')
        return None, mode
    else:
        if E_max == -1:  E_max = np.inf
        if event_no == -1:
            event_no = get_spe_key_no(eval_file, key="HT")
        n = 0

        if mode == 'gaussian_model_fit' or mode == 'qgmf':
            #        sigma_x = []
            #        sigma_y = []
            #        err_sigma_x = []
            #        err_sigma_y = []
            #        with open(eval_file,'r') as spe_file:
            #            for line in spe_file:
            #                if n >= event_no:    break
            #                sl = line.split()
            #                if sl[0] == "HT":
            #                    if E_min<float(sl[3]) and float(sl[3])<E_max:
            #                        n+=1
            #                        sigma_x.append(float(sl[4]))
            #                        sigma_y.append(float(sl[5]))
            #                        progress_bar.setValue(100.*n/event_no)
            #                        in_range = True
            #                    else:
            #                        in_range = False
            #                if sl[0] == "ER" and in_range:
            #                    err_sigma_x.append(float(sl[4]))
            #                    err_sigma_y.append(float(sl[5]))
            #        return [sigma_x,sigma_y],[err_sigma_x,err_sigma_y],mode
            return None, mode

        elif mode == 'four_px_area_clustering' or mode == 'four_px_area' or mode == 'clustering' or mode == 'gendreau' or mode == 'epic' or mode == 'cnn':
            get_px = False
            pxs = []
            s = 0
            data = []
            with open(eval_file, 'r') as spe_file:
                for line in spe_file:
                    if n >= event_no:    break
                    sl = line.split()
                    if sl[0] == "HT":
                        n_px = int(float(sl[4]))
                        adu = float(sl[3])
                        if len(pxs) > 0:
                            data.append([s, np.max(pxs) * s])
                            pxs = []
                        if E_min < adu and adu < E_max and n_min <= n_px and n_px <= n_max:
                            n += 1
                            s = float(sl[3])
                            get_px = True
                            if progress_bar:
                                progress_bar.setValue(100. * n / event_no)
                        else:
                            get_px = False
                    if sl[0] == "PX" and get_px:
                        pxs.append(float(sl[3]))
            return np.array(data), mode

        else:
            return None, mode


def get_photon_flux_evolution(eval_file, event_no, progress_bar=[]):
    mode = _get_mode(eval_file)
    if eval_file.split('.')[-1] != 'spe':
        print(f'WARNING: {eval_file.split(".")[-1]} file does not support photon flux evolution calculation.')
        return None, None
    else:
        if event_no == -1:
            event_no = get_spe_key_no(eval_file, key='HT')
        n = 0
        SF_list = []
        photon_flux_evolution = []

        with open(eval_file) as spe_file:
            for i, line in enumerate(spe_file):
                if n >= event_no:    break
                sl = line.split()
                if sl[0] == "SF":
                    SF_list.append(int(sl[1]))
                    photon_flux_evolution.append(int(sl[2]))
                    n += int(sl[2])
                if i % 100 == 0:
                    if progress_bar:
                        progress_bar.setValue(100. * n / event_no)
        return SF_list, photon_flux_evolution

#
# def read_data_cube(file_path):
#     """
#     Parameters
#     ----------
#     file_path : string
#         Direct or relative path to the file containing energetic and spatial
#         infromation of all photons.
#
#     Author: JB
#     """
#
#     from h5py import File
#     cubes = File(file_path, "r")
#     print('read', file_path)
#     return cubes


def save_photon_map(file_path, label, photon_map):
    mt.save_image_file('.'.join(file_path.split('.')[:-1]) + '_' + label + '_pm.tif', photon_map, dtype='int32')


def sum_spe_files(file_list, progress_bar=[]):
    #    def mysum_cubes(c1,c2):
    #        print('mysum_cubes:',np.shape(c1)
    #        for i in range(np.shape(c1)[0]):
    #            c1[i,:,:] = c1[i,:,:]+c2[i,:,:]
    #        return c1
    h5 = True
    spe = True
    for f in file_list:
        if f.split('.')[-1] != 'h5':    h5 = False
        if f.split('.')[-1] != 'spe':   spe = False

    if h5:
        try:
            os.mkdir('tmp')
        except:
            pass
        redundant_no = 2
        sum_h5 = 'tmp/sum_1.h5'
        while os.path.isfile(sum_h5):
            sum_h5 = 'tmp/sum_' + str(redundant_no) + '.h5'
            redundant_no += 1

        with h5py.File(sum_h5, 'w') as f:

            for cube in file_list[0]:
                f.create_dataset(cube, np.shape(file_list[0][cube]), dtype='uint32', compression="gzip")
            for i, cubes in enumerate(file_list):
                for j, cube in enumerate(f):
                    f[cube][...] = f[cube].value + cubes[cube].value
                    progress_bar.setValue(int(100. * (j + 1) / len(f)))
        return sum_h5

    elif spe:
        raise ValueError('summing spe spectra is not implemented, yet')

    else:
        raise ValueError('no uniform spectra file format')


def make_photon_maps(file_path, rois, event_no, progress_bar=[], n1=1, n2=-1):
    """
    Parameters
    ----------
    file_path : string
        Direct or relative path to the file containing energetic and spatial
        infromation of all photons.
    rois : dictionary

    Author: JB
    """

    photon_maps = {}
    if n2 == -1: n2 = np.inf

    if file_path.split('.')[-1] == 'h5':
        cubes = h5py.File(file_path, "r")
        n_x, n_y = np.shape(cubes[cubes.keys()[0]])[0] * 2, np.shape(cubes[cubes.keys()[0]])[1] * 2
        max_calc_steps1 = len(rois) * len(cubes)
        max_calc_steps2 = len(rois) * n_x

        r = 1
        c = 1
        n = 1
        for roi in rois:
            split_cubes = []
            photon_maps[roi] = np.zeros((n_x, n_y))
            for cube in cubes:
                if rois[roi]:
                    split_cubes.append(np.sum(cubes[cube][:, :, rois[roi][0]:rois[roi][1]], axis=2))
                else:
                    split_cubes.append(np.sum(cubes[cube], axis=2))
                if progress_bar:    progress_bar.setValue(
                    int(50. * r * c / max_calc_steps1 + 50. * r * n / max_calc_steps2))
                c += 1

            for x in range(n_x):
                for y in range(n_y):
                    if x % 2 == 0 and y % 2 == 0:
                        photon_maps[roi][x, y] = split_cubes[0][int(x / 2), int(y / 2)]
                    elif x % 2 == 1 and y % 2 == 0:
                        photon_maps[roi][x, y] = split_cubes[1][int(x / 2), int(y / 2)]
                    elif x % 2 == 0 and y % 2 == 1:
                        photon_maps[roi][x, y] = split_cubes[2][int(x / 2), int(y / 2)]
                    elif x % 2 == 1 and y % 2 == 1:
                        photon_maps[roi][x, y] = split_cubes[3][int(x / 2), int(y / 2)]
                try:
                    progress_bar.setValue(int(50. * r * c / max_calc_steps1 + 50. * r * n / max_calc_steps2))
                except:
                    pass
                try:
                    progress_bar.emit(
                        50. * r * c / max_calc_steps1 + 50. * r * n / max_calc_steps2)  # if it is a signal
                except:
                    pass
                n += 1
            r += 1
        return photon_maps

    if file_path.split('.')[-1] == 'spe':
        events = 0
        if event_no == -1:
            event_no = get_spe_key_no(file_path, key='HT')
        try:
            with open(file_path) as spe_file:
                for line in spe_file:
                    if line.split()[0] == "SI":
                        [x_max, y_max] = np.array(line.split(' ')[-2:], dtype='int')
                        print(x_max, y_max)
                        break
        except:
            raise IOError('cannot determine frame size from spe file')

        for roi in rois:
            photon_maps[roi] = np.zeros((x_max, y_max))

        spe_file = open(file_path)
        for i, line in enumerate(spe_file):
            if events >= event_no:    break
            try:
                sl = line.split()
                if sl[0] == "HT":
                    x = int(float(sl[1]))
                    y = int(float(sl[2]))
                    counts = float(sl[3])
                    n = int(float(sl[4]))
                    events += 1

                    for roi in rois:
                        try:
                            if rois[roi][0] < counts and counts < rois[roi][1]:
                                if n1 <= n and n <= n2:
                                    photon_maps[roi][x, y] += 1
                        except:
                            print(ValueError(
                                'line skipped: rois wrongly defined (use {roi_name : [low,high]}) or photon map overflow'))
            except:
                print(ValueError('line skipped: cannot handle values in:\n\t' + line))
            if (events % (int(event_no / 100))) == 0:
                if progress_bar:
                    try:
                        progress_bar.setValue(int(100. * events / event_no))  # if it is progress_bar
                    except:
                        pass
                    try:
                        progress_bar.emit(int(100. * events / event_no))  # if it is a signal
                    except:
                        pass
        spe_file.close()
        for pm in photon_maps:
            print(np.sum(photon_maps[pm]))
        return photon_maps
