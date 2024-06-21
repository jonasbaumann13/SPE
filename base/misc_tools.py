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

def sort_files(file_list, mode='created', fun=None):
    """
    sorts the file paths in a list depending on a specific keyword

    Parameter
    ---------
    file_list : list of strings
        absolte or relative paths to files
    mode: string
        keyword for mode of sort algorithm, see 'supported modes' for details

    Return
    ------
    file_list : 2d list
        sorted list of file paths and sorted key as second list entry

    supported modes
    ---------------
    'created' : sorted after time stamp of creation of file

    Author
    ------
    Jonas Baumann
    """

    import numpy as np
    import os
    if mode == 'created':
        created = []
        for f in file_list:
            created.append(os.path.getmtime(f))
        comb_list = np.array([file_list, created])
        # return comb_list[comb_list[:,1].argsort()][0],comb_list[comb_list[:,1].argsort()][1]
        return list(zip(*sorted(comb_list.T, key=lambda x: float(x[1]))))
    if mode == 'number':
        num = []
        for i in range(len(file_list)):
            f_name = file_list[i]
            f_name = f_name.split(os.sep)[-1].split('.')[-2]
            if fun is None:
                num.append(float(f_name))
            else:
                num.append(fun(f_name))
        comb_list = np.array([file_list, num])
        return list(zip(*sorted(comb_list.T, key=lambda x: float(x[1]))))


def mkdir(path: str):
    """
    creates the given path without exception in case it exists.

    Parameters
    ----------
    path : str
        path to be created

    Returns
    -------
    None.

    """
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


def median_standard_deviation(arr, axis):
    """
    calculates and returns the median standard deviation, which is the square root
    of the median of the squared deviations.

    Parameters
    ----------
    arr : array like
        input array.
    axis : int
        axis to perform calculation.

    Returns
    -------
    median_std : np.array
        median standard deviation.
    median : np.array
        median of arr using np.median(arr, axis=axis).

    """
    median = np.median(arr, axis=axis)
    sq_dev = (arr-median)**2
    median_std = np.sqrt(np.median(sq_dev, axis=axis))
    return median_std, median



class File(object):
    def __init__(self, fname, dtype=np.float32):
        self._dtype = dtype
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.int16)[0])
        self._ydim = np.int64(self.read_at(656, 1, np.int16)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, np.int8)
        rawtime = self.read_at(172, 6, np.int8)
        strdate = ''
        for ch in rawdate :
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate,"%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim, self._dtype)
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        self._fid.close()

class sydor_raw_file(object):
    def __init__(self, fname, xdim, ydim, dtype=np.float32):
        self._dtype = dtype
        self._xdim = xdim
        self._ydim = ydim
        self._fid = open(fname, 'rb')

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self, number):
        img = self.read_at(2*number*self._xdim*self._ydim, self._xdim*self._ydim, self._dtype)
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        self._fid.close()


def load_image_file(file_name, xydim=None, dtype=None, number=1):
    """
    load numpy array from image data

    Parameters
    ----------
    file_name : string
        full path to image file

    Returns
    -------
    frame : 2d numpy array
        the loaded data
    xydim : tuple
        x and y dimension of image, used for loading .raw data
    dtype : data type in .raw data

    Author
    ------
    Jonas Baumann
    """

    from numpy import loadtxt
    from imageio.v2 import imread
    import pandas as pd

    ftype = file_name.split('.')[-1].lower()
    if ftype == 'txt' or ftype == 'dat' or ftype == 'tsv':
        try:
            frame = pd.read_csv(file_name)  # load the txt
        except:
            pass
        else:
            frame = loadtxt(file_name)
    elif ftype == 'tif' or ftype == 'tiff' or ftype == 'png':
        frame = imread(file_name)
    elif ftype == 'spe':
        spe = File(file_name)
        frame = spe.load_img()
        spe.close()
    elif ftype == 'raw':
        frames = []
        raw = sydor_raw_file(file_name, *xydim, dtype=dtype)
        for i in range(number):
            frames.append(raw.load_img(i))
        raw.close()
        if len(frames) == 1:
            frame = frames[0]
        else:
            frame = np.array(frames)
    else:
        raise FileNotFoundError('wrong file format. please use txt, dat, tsv, tif or spe')
    return frame


def save_image_file(file_path, frame, dtype=None):
    """
    saves an image from a numpy 2d-array

    Parameters
    ----------
    file_path : string
        path and name of file to save
    frame : numpy.array (2d)
        data array to transform to image
    dtype : string, optional
        determination of data type. Supported are e.g. 'int16', 'uint32', float ...,
        default is None, then frame determines the dtype
    Author
    ------
    Jonas Baumann
    """

    from numpy import savetxt, array as nparr
    from . import localtifffile as tifff

    ftype = file_path[-3:]
    if ftype == 'txt' or ftype == 'dat' or ftype == 'tsv':
        savetxt(file_path, frame)  # save the txt
    elif ftype == 'tif':
        frame = nparr(frame, dtype=dtype)
        tifff.imsave(file_path, frame, dtype=dtype)
    else:
        raise IOError('wrong file format. please use txt, dat, tsv or tif')


def std_pic(folder, file_list=0, name="dark", parts=1, file_format="txt", limit=0, progress_bar=[]):
    """
    std_pic creates a frame with the standard deviation of each pixel out of several dark frames.

    Parameters
    ----------
    folder : string
        Folder, where the dark frames are located.

    file_list: list of strings, optional
        If a file list is given, it will be used instead of trying to locate the
        dark frames. Must only contain dark frames.

    name: string
        Keyword to identify dark frames when the folder is searched. Default is "dark".

    parts: integer
        Number of parts, in which the frames will be split to save memory space. Default is 1.

    file_format: string
        File format of the dark frames. txt, dat, tsv and tif are supported. Default is txt.

    limit: float
        median+(median-minimum)xlimit determines the threshold, above which single values of a pixel are replaced by the median
        value. For the default of limit = 0, no threshold is set.

    progress_bar : QtGui.QProgressBar, optional
        to show the progress in a GUI

    Returns
    -------
    median_frame : 2d numpy array
        Frame containing the pixelwise median of each dark image.

    std_frame : 2d numpy array
        Frame containing the standard deviations of each pixel.

    replacement_list : 2d list
        List of the pixel values have been replaced by the limit process described above. First column is an array
        containing the first pixel coordinate, second column the second. The third item is a list with the according
        dark frames.

    mean_frame : 2d numpy array
        Frame containing the mean value of each pixel.

    Author
    ------
    Steffen Staeck
    """

    import glob
    import numpy as np
    try:
        import two_d as td
    except:
        import axp_tools.two_d as td

    def replace_cosmics(data, limit):  # replace the values above the threshold

        data = np.array(data, dtype=float)
        median = np.array(np.median(data, axis=0), dtype=float)
        minimum = np.array(np.min(data, axis=0), dtype=float)
        data_tmp = (data - median) / (np.abs((np.median(median) - np.median(
            minimum))) + median - minimum)  # vorderer Teil global, hinteren beiden Argumente pixelweise
        repl_list = [[] for a in range(3)]
        for i, d in enumerate(data_tmp):
            replaced = np.where(d > limit)
            if len(replaced[0]) > 0:
                repl_list[0] = np.concatenate((repl_list[0], replaced[0]))
                repl_list[1] = np.concatenate((repl_list[1], replaced[1]))
                repl_list[2] = np.concatenate((repl_list[2], [i for l in range(len(replaced[0]))]))
            for (x, y) in np.array(replaced).T:
                data[i][x, y] = median[x, y]

        return data, repl_list

    def concatenate_lists(dark_file_arr_part, dark_file_arr_median, dark_file_arr_mean, dark_file_arr, limit, part):

        if limit != 0:  # add to replacment list
            dark_file_arr_part, replacement_list_add = replace_cosmics(dark_file_arr_part, limit)
            if list(replacement_list_add[0]):   replacement_list_add[0] += part * i
            for el in range(3): replacement_list[el] = np.concatenate((replacement_list[el], replacement_list_add[el]))
        dark_file_arr_median = np.concatenate((dark_file_arr_median, np.median(dark_file_arr_part, axis=0)),
                                              axis=0)  # add to std array
        dark_file_arr_mean = np.concatenate((dark_file_arr_mean, np.mean(dark_file_arr_part, axis=0)), axis=0)
        dark_file_arr = np.concatenate((dark_file_arr, np.std(dark_file_arr_part, axis=0)), axis=0)  # add to std array

        return dark_file_arr_median, dark_file_arr_mean, dark_file_arr, replacement_list

    def make_slices(name, file_list, i, parts,
                    progress_bar=[]):  # Loads the images and slices them depending on parts, i denotes the part.

        j = 0  # j denotes the frame
        dark_file_arr_part = []
        dark_file_mean_arr = []
        for file_name in file_list:  # load every frame containing the keyword "name" in file_list

            dark_frame = td.load_image_file(file_name)
            shape = np.shape(dark_frame)[0]
            part = int(shape / parts)
            try:
                dark_frame_part = np.array(dark_frame[int(part * i):int(part * (i + 1))])
            except:  # if i is too big, it will take the remaining the slice. This way with i = parts, the remainder of the frame can be computed.
                dark_frame_part = np.array(dark_frame[part * i:])
            if i == 0:  dark_file_mean_arr.append(
                np.mean(dark_frame))  # Compute the mean array. For efficiency reasons, only if i==0.
            del dark_frame

            if shape % parts != 0:
                end = parts + 1
            else:
                end = parts
            if progress_bar:
                progress_bar.setValue(100 * (i * len(file_list) + j + 1) / (len(file_list) * end))
            else:
                print("part=", i, "image=", j, ':', 100 * (i * len(file_list) + j + 1) / (len(file_list) * end), '%')

            dark_file_arr_part.append(dark_frame_part)  # start putting together the array for calculating the stds
            j += 1

        return dark_file_arr_part, dark_file_mean_arr, shape

    dark_file_arr = []
    dark_file_mean_arr = []
    replacement_list = []

    if file_list == 0:
        file_list = glob.glob(
            folder + "*" + name + "*." + file_format)  # gather frames, now filter directly for keyword.
        file_list = [a.replace('\\', '/') for a in file_list]
        print(file_list)

    if parts == 1:  # algorithm if framespart are not divided
        dark_file_arr, dark_file_mean_arr, shape = make_slices(name, file_list, i=0, parts=parts,
                                                               progress_bar=progress_bar)

        if limit != 0:
            dark_file_arr, replacement_list = replace_cosmics(dark_file_arr, limit)
        median_frame = np.median(dark_file_arr, axis=0)
        mean_frame = np.mean(dark_file_arr, axis=0)
        std_frame = np.std(dark_file_arr, axis=0)

    else:  # algorithm for divided frames

        for i in range(parts):  # i denotes the part of the frame
            if i == 0:
                dark_file_arr_part, dark_file_mean_arr, shape = make_slices(name, file_list, i, parts=parts,
                                                                            progress_bar=progress_bar)
            else:
                dark_file_arr_part = make_slices(name, file_list, i, parts, progress_bar=progress_bar)[0]

            if i == 0:  # initialize the std array and replacement list
                if limit != 0:
                    dark_file_arr_part, replacement_list = replace_cosmics(dark_file_arr_part, limit)
                dark_file_arr_median = np.median(dark_file_arr_part, axis=0)
                dark_file_arr_mean = np.mean(dark_file_arr_part, axis=0)
                dark_file_arr = np.std(dark_file_arr_part, axis=0)

            else:
                part = shape / parts
                dark_file_arr_median, dark_file_arr_mean, dark_file_arr, replacement_list = concatenate_lists(
                    dark_file_arr_part, dark_file_arr_median, dark_file_arr_mean, dark_file_arr, limit, part)

        if shape % parts != 0:  # Calculate the remaining part of the frame, if there is one.
            dark_file_arr_part = make_slices(name, file_list, i=parts, parts=parts, progress_bar=progress_bar)[0]
            dark_file_arr_median, dark_file_arr_mean, dark_file_arr, replacement_list = concatenate_lists(
                dark_file_arr_part, dark_file_arr_median, dark_file_arr_mean, dark_file_arr, limit, part)

        median_frame = dark_file_arr_median
        std_frame = dark_file_arr
        mean_frame = dark_file_arr_mean

    std_frame[std_frame == 0] = np.mean(std_frame)
    return median_frame, std_frame, replacement_list, dark_file_mean_arr, mean_frame



def image_line(angle,nx,ny):
    m = np.tan(np.deg2rad(angle))
    n = ny/2.-m*nx/2.
    return lambda x: m*x+n
