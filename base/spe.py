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
from numba import jit
import os, inspect
try:
    import torch
except:
    print('pytorch not installed. Neural networks not available')
    
@jit(nopython=True, cache=False)
def bytes_equal(a, b):
    if len(a) != len(b):
        return False
    for char1, char2 in zip(a, b):
        if char1 != char2:
            return False
    return True


@jit(nopython=True, cache=False)
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


@jit(nopython=True, cache=False)
def set_px(x,y,sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image,cluster_nr):
    tmp_s = 0.
    tmp_x = 0.
    tmp_y = 0.
    tmp_n = 0
    px_int = [1.]
    del px_int[:]
    px_pos = [(1,2)]
    del px_pos[:]
    if 0 <= x and x < x_max and 0 <= y and y < y_max:
        if image_copy[x][y] >= sig_fac_2*sigma_image[x][y]:
            tmp_s += image_copy[x][y]
            tmp_x += x*tmp_s
            tmp_y += y*tmp_s
            tmp_n += 1
            px_int.append(image_copy[x][y])
            px_pos.append((x,y))
            image_copy[x][y] = 0
            cluster_image[x][y] = cluster_nr
            next_px = find_adjacent_pixels(x,y,cluster_nr,sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image)
            tmp_s += next_px[0]
            tmp_x += next_px[1]
            tmp_y += next_px[2]
            tmp_n += next_px[3]
            px_int += next_px[4]
            px_pos += next_px[5]
    return tmp_s,tmp_x,tmp_y,tmp_n,px_int,px_pos


@jit(nopython=True, cache=False)
def find_adjacent_pixels(x,y,cluster_nr,sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image):
    tmp_s = 0.
    tmp_x = 0.
    tmp_y = 0.
    tmp_n = 0
    px_int = [1.]
    del px_int[:]
    px_pos = [(1,2)]
    del px_pos[:]
    pos_arr = [(0,0),(-1,0),(1,0),(0,1),(0,-1)]
    for pos in pos_arr:
        next_px = set_px(x+pos[0],y+pos[1],sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image,cluster_nr)
        tmp_s += next_px[0]
        tmp_x += next_px[1]
        tmp_y += next_px[2]
        tmp_n += next_px[3]
        px_int += next_px[4]
        px_pos += next_px[5]
    return tmp_s,tmp_x,tmp_y,tmp_n,px_int,px_pos


def get_significant_pixels(image, sigma_image, sig_fac_1, defect_map):
    sig_fac_1 = float(sig_fac_1)
    if defect_map is None:
        significant_pixels = np.where(image >= sigma_image*sig_fac_1)
    else:
        if np.shape(image) == np.shape(defect_map):
            significant_pixels = np.where(np.logical_and(image >= sigma_image*sig_fac_1, defect_map==0))
        else:
            raise IOError('shape of defect_map and image do not match')
    return significant_pixels    

            
##########################
#####   CLUSTERING   #####
def clustering(image,sig_fac_1,sig_fac_2,sigma,ret_cluster_image=False,defect_map=None):
    """
    Evaluate single phton events on a 2d array (e.g. ccd, cmos) using the
    clustering method. For reference see dissertation of Jonas Baumann e.g. @
    http://dx.doi.org/10.14279/depositonce-6581

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    sig_fac_2 : float
        second noise threshold factor. determines which pixels surrounding the
        cluster seed are added to the cluster.
    sigma : float or 2d-array
        expected standard deviation of the noise in the image, if 2d array is
        given, then the standard deviation is used pixelwise
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.
    defect_map : numpy 2d array, optional
        n1xn2 array with zeros in pixels, which should be used for spe algorithm
        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Jonas Baumann
    """
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    image_copy = np.array(image)
    photon_events = clustering_jit(image,image_copy,sig_fac_2,sigma_image,x_max,y_max, significant_x_values,significant_y_values)
    if ret_cluster_image:
        return list(map(list, zip(*photon_events[:-1]))), photon_events[-1]
    else:
        return list(map(list, zip(*photon_events[:-1])))


@jit(nopython=True, cache=False)
def clustering_jit(image,image_copy,sig_fac_2,sigma_image,x_max,y_max,significant_x_values,significant_y_values):
#                
    cluster_image = np.zeros((x_max,y_max))
    cluster_nr = 0
    px_x = []
    px_y = []
    px_I = []
    px_n = []
    px_I_rel = []
    px_Pos_rel = []
#    
    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    
        if cluster_image[x][y] == 0 and contains_box(x,y,x_max,y_max,3):
            s = image[x-1:x+2,y-1:y+2]
            if image[x][y] == np.max(s):
                image_copy[x][y] = 0

                tmp_event_sum = image[x][y]
                tmp_event_x = x*tmp_event_sum
                tmp_event_y = y*tmp_event_sum
                tmp_event_n = 1
                px_list_int = [image[x][y]]
                px_list_pos = [(x,y)]

                cluster_nr += 1
                cluster_image[x][y] = cluster_nr
#                
                pos_arr = [(-1,-1),(0,-1),(-1,0),(0,0)]
                for pos in pos_arr:
                    next_px = find_adjacent_pixels(x+pos[0],y+pos[1],cluster_nr,sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image)
                    tmp_event_sum += next_px[0]
                    tmp_event_x += next_px[1]
                    tmp_event_y += next_px[2]
                    tmp_event_n += next_px[3]
                    px_list_int += next_px[4]
                    px_list_pos += next_px[5]
                px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                px_I.append(tmp_event_sum)
                px_n.append(tmp_event_n)
                px_I_rel.append(px_list_int)
                px_Pos_rel.append(px_list_pos)

    return px_x, px_y, px_I, px_n, px_I_rel, px_Pos_rel, cluster_image


##########################
#####   4px-AREA   #######
def four_px_area(image,sig_fac_1,sigma,ret_cluster_image=False,defect_map=None):
    """
    Evaluate single phton events on a 2d array (e.g. ccd, cmos) using the
    four_px_area method. For reference see dissertation of Jonas Baumann e.g. @
    http://dx.doi.org/10.14279/depositonce-6581

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    sigma : float or 2d-array
        expected standard deviation of the noise in the image, if 2d array is
        given, then the standard deviation is used pixelwise
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.
    defect_map : numpy 2d array, optional
        n1xn2 array with zeros in pixels, which should be used for spe algorithm
        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Jonas Baumann
    """
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    image_copy = np.array(image)
    photon_events = four_px_area_jit(image,image_copy,sigma_image,x_max,y_max, significant_x_values,significant_y_values)
    if ret_cluster_image:
        return list(map(list, zip(*photon_events[:-1]))), photon_events[-1]
    else:
        return list(map(list, zip(*photon_events[:-1])))


from matplotlib import pyplot as plt

@jit(nopython=True, cache=False)
def four_px_area_jit(image,image_copy,sigma_image,x_max,y_max,significant_x_values,significant_y_values):
    cluster_image = np.zeros((x_max,y_max))
    cluster_nr = 0
    xmesh = np.array([[0, 0], [1, 1]])
    ymesh = np.array([[0, 1], [0, 1]])
    px_x = []
    px_y = []
    px_I = []
    px_n = []
    px_I_rel = []
    px_Pos_rel = []

    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    
        if cluster_image[x][y] == 0 and contains_box(x,y,x_max,y_max,3):
            s = image[x-1:x+2,y-1:y+2]
            if image[x][y] == np.max(s):
                sub_s = []
                sub_s_sum = []
                pos_arr = [(0,0),(0,1),(1,0),(1,1)]
                for pos in pos_arr:
                    sub_s.append(s[pos[0]:pos[0]+2,pos[1]:pos[1]+2])
                    sub_s_sum.append(np.sum(sub_s[-1]))
                
                ind = np.where(sub_s_sum == np.max(np.array(sub_s_sum)))[0][0]
                #ind = sub_s_sum.index(np.max(np.array(sub_s_sum)))

                px_list_pos = [(x-1+pos_arr[ind][0],y-1+pos_arr[ind][1]),(x+pos_arr[ind][0],y-1+pos_arr[ind][1]),(x-1+pos_arr[ind][0],y+pos_arr[ind][1]),(x+pos_arr[ind][0],y+pos_arr[ind][1])]
                tmp_event_sum = sub_s_sum[ind]
                tmp_event_x = np.sum((xmesh+x-1+pos_arr[ind][0])*sub_s[ind])
                tmp_event_y = np.sum((ymesh+y-1+pos_arr[ind][1])*sub_s[ind])
                px_list_int = list(np.ravel(sub_s[ind].T))
                cluster_nr += 1
                cluster_image[x-1+pos_arr[ind][0]:x+1+pos_arr[ind][0],y-1+pos_arr[ind][1]:y+1+pos_arr[ind][1]] = cluster_nr                     # set the 2x2 area in the cluster image to the cluster number
                image_copy[x-1+pos_arr[ind][0]:x+1+pos_arr[ind][0],y-1+pos_arr[ind][1]:y+1+pos_arr[ind][1]] = 0                                 # set the 2x2 area in the image_copy to zero
                tmp_event_n = 4
                
                if tmp_event_sum > 0:
                    px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                    px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                else:
                    px_x.append(0)
                    px_y.append(0)
                    #print('SPE WARNING: negative event intensity detected. Check noise threshold and / or master dark!')
                px_I.append(tmp_event_sum)
                px_n.append(tmp_event_n)
                px_I_rel.append(px_list_int)
                px_Pos_rel.append(px_list_pos)

    return px_x, px_y, px_I, px_n, px_I_rel, px_Pos_rel, cluster_image

#####################################
#####   4px-AREA-CLUSTERING   #######
def four_px_area_clustering(image,sig_fac_1,sig_fac_2,sigma,ret_cluster_image=False,defect_map=None):
    """
    Evaluate single phton events on a 2d array (e.g. ccd, cmos) using the
    four_px_area_clustering method. For reference see dissertation of Jonas Baumann e.g. @
    http://dx.doi.org/10.14279/depositonce-6581

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    sig_fac_2 : float
        second noise threshold factor. determines which pixels surrounding the
        cluster seed are added to the cluster.
    sigma : float or 2d-array
        expected standard deviation of the noise in the image, if 2d array is
        given, then the standard deviation is used pixelwise
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.
    defect_map : numpy 2d array, optional
        n1xn2 array with zeros in pixels, which should be used for spe algorithm
        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Jonas Baumann
    """
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    image_copy = np.array(image)
    photon_events = four_px_area_clustering_jit(image,image_copy,sig_fac_2,sigma_image,x_max,y_max, significant_x_values,significant_y_values)
    if ret_cluster_image:
        return list(map(list, zip(*photon_events[:-1]))), photon_events[-1]
    else:
        return list(map(list, zip(*photon_events[:-1])))


@jit(nopython=True, cache=False)
def four_px_area_clustering_jit(image,image_copy,sig_fac_2,sigma_image,x_max,y_max,significant_x_values,significant_y_values):
#                
    cluster_image = np.zeros((x_max,y_max))
    cluster_nr = 0
    xmesh = np.array([[0, 0], [1, 1]])
    ymesh = np.array([[0, 1], [0, 1]])
    px_x = []
    px_y = []
    px_I = []
    px_n = []
    px_I_rel = []
    px_Pos_rel = []
#    
    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    
        if cluster_image[x][y] == 0 and contains_box(x,y,x_max,y_max,3):
            s = image[x-1:x+2,y-1:y+2]
            if image[x][y] == np.max(s):
                sub_s = []
                sub_s_sum = []
                pos_arr = [(0,0),(0,1),(1,0),(1,1)]
                for pos in pos_arr:
                    sub_s.append(s[pos[0]:pos[0]+2,pos[1]:pos[1]+2])
                    sub_s_sum.append(np.sum(sub_s[-1]))
                
                ind = np.where(sub_s_sum == np.max(np.array(sub_s_sum)))[0][0]
                # ind = sub_s_sum.index(np.max(np.array(sub_s_sum)))
                
                px_list_pos = [(x-1+pos_arr[ind][0],y-1+pos_arr[ind][1]),(x+pos_arr[ind][0],y-1+pos_arr[ind][1]),(x-1+pos_arr[ind][0],y+pos_arr[ind][1]),(x+pos_arr[ind][0],y+pos_arr[ind][1])]
                tmp_event_sum = sub_s_sum[ind]
                tmp_event_x = np.sum((xmesh+x-1+pos_arr[ind][0])*sub_s[ind])
                tmp_event_y = np.sum((ymesh+y-1+pos_arr[ind][1])*sub_s[ind])
                
                px_list_int = list(np.ravel(sub_s[ind].T))
                
                cluster_nr += 1
                cluster_image[x-1+pos_arr[ind][0]:x+1+pos_arr[ind][0],y-1+pos_arr[ind][1]:y+1+pos_arr[ind][1]] = cluster_nr                     # set the 2x2 area in the cluster image to the cluster number
                image_copy[x-1+pos_arr[ind][0]:x+1+pos_arr[ind][0],y-1+pos_arr[ind][1]:y+1+pos_arr[ind][1]] = 0                                 # set the 2x2 area in the image_copy to zero
                tmp_event_n = 4
                
                # now the further clustering
                nonzero_pos_sa2x2 = np.where(sub_s[ind] >= 0)                 # should be all 4 positions
                for i in range(len(nonzero_pos_sa2x2[0])):                                          # check in each of the (former) nonzero pixels of the 2x2 area for further adjacent pixels
                    next_px = find_adjacent_pixels(x-1+nonzero_pos_sa2x2[0][i]+pos_arr[ind][0],y-1+nonzero_pos_sa2x2[1][i]+pos_arr[ind][1],cluster_nr,sig_fac_2,x_max,y_max,sigma_image,image_copy,cluster_image)
                    tmp_event_sum += next_px[0]
                    tmp_event_x += next_px[1]
                    tmp_event_y += next_px[2]
                    tmp_event_n += next_px[3]
                    px_list_int += next_px[4]
                    px_list_pos += next_px[5]
                
                if tmp_event_sum > 0:
                    px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                    px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                else:
                    px_x.append(0)
                    px_y.append(0)
                # px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                # px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                px_I.append(tmp_event_sum)
                px_n.append(tmp_event_n)
                px_I_rel.append(px_list_int)
                px_Pos_rel.append(px_list_pos)

    return px_x, px_y, px_I, px_n, px_I_rel, px_Pos_rel, cluster_image


######################
#####   ASCA   #######
def asca(image,sig_fac_1,sig_fac_2,sigma,ret_cluster_image=False,defect_map=None):
    """
    Evaluate single phton events on a 2d array (e.g. ccd, cmos) using the
    method by Gendreau from his 1995 dissertation.

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    sig_fac_2 : float
        second noise threshold factor. determines which pixels inside the
        3x3 cluster seed are considered in the evaluation.
    sigma : float
        expected standard deviation of the noise in the image
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.

        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Jonas Baumann
    """
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    sigma_diff = image/np.array(sigma_image,dtype='float')              # to ensure no rounding
    photon_events = asca_jit(image, sigma_diff, sig_fac_2, sigma_image, x_max, y_max, significant_x_values, significant_y_values)
    if ret_cluster_image:
        return list(map(list, zip(*photon_events[:-1]))), photon_events[-1]
    else:
        return list(map(list, zip(*photon_events[:-1])))
    

@jit(nopython=True, cache=False)
def asca_jit(image, sigma_diff, sig_fac_2, sigma_image, x_max, y_max, significant_x_values, significant_y_values):
               
    cluster_image = np.zeros((x_max,y_max))
    cluster_nr = 0
#    categories = ['Single', 'Single+', 'Up', 'Down', 'Left', 'Right', 'Single Sided+', 'L+Q', 'Other']
    count_list = np.zeros(9)

    px_x = []
    px_y = []
    px_I = []
    px_n = []
    px_I_rel = []
    px_Pos_rel = []

    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    
        if cluster_image[x][y] == 0 and contains_box(x,y,x_max,y_max,3):
            discard = False

            s = image[x-1:x+2,y-1:y+2]
            if image[x][y] == np.max(s):
                sigma_diff[x][y] = 0
                
                tmp_event_sum = image[x][y]
                tmp_event_x = x*tmp_event_sum
                tmp_event_y = y*tmp_event_sum
                tmp_event_n = 1
                px_list_int = [image[x][y]]
                px_list_pos = [(x,y)]

                discard = False

                scp = np.where(sigma_diff[x-1:x+2,y-1:y+2]>=sig_fac_2) # cluster pixel above threshold
                if len(np.where(scp[0] == 1)[0]) == 0 and len(np.where(scp[1] == 1)[0]) == 0:   #get rid of single+
                    if len(np.where(scp[0] != 1)[0]) != 0 or len(np.where(scp[1] != 1)[0]) !=0:
                        discard = True
                        category = 1
                    else:
                        category = 0
                if len(np.where(scp[0] != 1)[0]) + len(np.where(scp[1] != 1)[0]) > 5:
                    category = 8
                    discard = True             
                    
                if len(np.where(scp[0] == 1)[0]) == 1 or len(np.where(scp[1] == 1)[0]) == 1:
                    if len(np.where(scp[0] != 1)[0]) + len(np.where(scp[1] != 1)[0]) > 5:
                        discard = True   #more than two corner pixels above threshold
                        category = 6
                    if len(np.where(scp[0] == 1)[0]) == len(np.where(scp[1] == 1)[0]):
                        discard = True   #take care of L+Q
                        category = 7
                if len(np.where(scp[0] == 1)[0]) == 2 or len(np.where(scp[1] == 1)[0]) == 2: 
                    discard = True   #get rid of three in a row
                    category = 8

                            
                if discard == False:
                    if len(np.where(scp[0] == 1)[0]) == 1: #deal with single sided+ vertical
                        if sigma_diff[x,y+1] >=sig_fac_2:
                            if sigma_diff[x-1,y+1] >=sig_fac_2 or sigma_diff[x+1,y+1] >=sig_fac_2:
                                discard = True
                                category = 6
                                if sigma_diff[x-1,y+1] >=sig_fac_2 and sigma_diff[x+1,y+1] >=sig_fac_2:
                                    discard = True
                                    category = 8
                            else: 
                                category = 5
                                tmp_event_sum += image[x][y+1]
                                tmp_event_x += x*image[x][y+1]
                                tmp_event_y += (y+1)*image[x][y+1]
                                tmp_event_n += 1
                                px_list_int += [image[x][y+1]]
                                px_list_pos += [(x,y+1)]
                                sigma_diff[x][y+1] = 0
                                cluster_nr += 1
                                cluster_image[x][y+1] = cluster_nr
                                
                        if sigma_diff[x,y-1] >=sig_fac_2:
                            if sigma_diff[x-1,y-1] >=sig_fac_2 or sigma_diff[x+1,y-1] >=sig_fac_2:
                                discard = True
                                category = 6
                                if sigma_diff[x-1,y-1] >=sig_fac_2 and sigma_diff[x+1,y-1] >=sig_fac_2:
                                    discard = True
                                    category = 8#categories.index('Other')
                            else:
                                category = 4#categories.index('Left')
                                tmp_event_sum += image[x][y-1]
                                tmp_event_x += x*image[x][y-1]
                                tmp_event_y += (y+1)*image[x][y-1]
                                tmp_event_n += 1
                                px_list_int += [image[x][y-1]]
                                px_list_pos += [(x,y-1)]
                                sigma_diff[x][y-1] = 0
                                cluster_nr += 1
                                cluster_image[x][y-1] = cluster_nr
                                
                    if len(np.where(scp[1] == 1)[0]) == 1: #deal with single sided+ horizontal
                        if sigma_diff[x+1,y] >=sig_fac_2:
                            if sigma_diff[x+1,y+1] >=sig_fac_2 or sigma_diff[x+1,y-1] >=sig_fac_2:
                                discard = True
                                category = 6#categories.index('Single Sided+')
                                if sigma_diff[x+1,y+1] >=sig_fac_2 and sigma_diff[x+1,y-1] >=sig_fac_2:
                                    discard = True
                                    category = 8#categories.index('Other')
                            else:
                                category = 3#categories.index('Down')
                                tmp_event_sum += image[x+1][y]
                                tmp_event_x += (x+1)*image[x+1][y]
                                tmp_event_y += y*image[x+1][y]
                                tmp_event_n += 1
                                px_list_int += [image[x+1][y]]
                                px_list_pos += [(x+1,y)]
                                sigma_diff[x+1][y] = 0
                                cluster_nr += 1
                                cluster_image[x+1][y] = cluster_nr
                                
                        if sigma_diff[x-1,y] >=sig_fac_2:
                            if sigma_diff[x-1,y+1] >=sig_fac_2 or sigma_diff[x-1,y-1] >=sig_fac_2:
                                discard = True
                                category = 6#categories.index('Single Sided+')
                                if sigma_diff[x-1,y+1] >=sig_fac_2 and sigma_diff[x-1,y-1] >=sig_fac_2:
                                    discard = True
                                    category = 8#categories.index('Other')
                            else:
                                category = 2#categories.index('Up')
                                tmp_event_sum += image[x-1][y]
                                tmp_event_x += (x-1)*image[x-1][y]
                                tmp_event_y += y*image[x-1][y]
                                tmp_event_n += 1
                                px_list_int += [image[x-1][y]]
                                px_list_pos += [(x-1,y)]
                                sigma_diff[x-1][y] = 0
                                cluster_nr += 1
                                cluster_image[x-1][y] = cluster_nr
                
                icx,icy = np.where(sigma_diff[x-1:x+2,y-1:y+2]>=sig_fac_2)
                for x,y in zip(icx,icy):
                    sigma_diff[x,y] = 0
                                
                count_list[category] += 1
                if discard == False:
                    px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                    px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                    px_I.append(tmp_event_sum)
                    px_n.append(tmp_event_n)
                    px_I_rel.append(px_list_int)
                    px_Pos_rel.append(px_list_pos)
                    if category == 0:#categories.index('Single'):
                        cluster_nr += 1
                    cluster_image[x][y] = cluster_nr

    return px_x, px_y, px_I, px_n, px_I_rel, px_Pos_rel, cluster_image


######################
#####   EPIC   #######
def epic(image,sig_fac_1,sig_fac_2,sigma,ret_cluster_image=False,defect_map=None):
    """
    Evaluate single photon events on a 2d array (e.g. ccd, cmos) using the
    method presented in the European Photon Imaging Camera (EPIC) paper from 2001 by Turner et al.
    https://doi.org/10.1051/0004-6361:20000087. Note: Only X-ray patterns according to paper (0-12 + 31, which means
    events consisting of 1 to 4 pixels and 9 pixel events) are used. Suited for small charge cloud sizes only!

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    sig_fac_2 : float
        second noise threshold factor. determines which pixels inside the
        3x3 cluster seed are considered in the evaluation.
    sigma : float
        expected standard deviation of the noise in the image
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.

        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Steffen Staeck, adapted for numba by Jonas Baumann
    """
    
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    sigma_diff = image/np.array(sigma_image,dtype='float')              # to ensure no rounding
    photon_events = epic_jit(image, sigma_diff, sig_fac_2, sigma_image, x_max, y_max, significant_x_values, significant_y_values)
    if ret_cluster_image:
        return list(map(list, zip(*photon_events[:-1]))), photon_events[-1]
    else:
        return list(map(list, zip(*photon_events[:-1])))



@jit(nopython=True, cache=False)
def epic_jit(image, sigma_diff, sig_fac_2, sigma_image, x_max, y_max, significant_x_values, significant_y_values):
               
    cluster_image = np.zeros((x_max,y_max))
    cluster_nr = 0
    category = 9
#    ["Single", "Two", "Two+", "Three", "Three+", "Four", "Four+", "All", "Whatever", "Not evaluated"]
    count_list = np.zeros(10)

    px_x = []
    px_y = []
    px_I = []
    px_n = []
    px_I_rel = []
    px_Pos_rel = []

    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    
        if cluster_image[x][y] == 0 and contains_box(x,y,x_max,y_max,5):
            discard = False

            s = image[x-1:x+2,y-1:y+2]
            if image[x][y] == np.max(s):
                sigma_diff[x][y] = 0
                
                tmp_event_sum = image[x][y]
                tmp_event_x = x*tmp_event_sum
                tmp_event_y = y*tmp_event_sum
                tmp_event_n = 1
                px_list_int = [image[x][y]]
                px_list_pos = [(x,y)]

                discard = False

                scp = np.where(sigma_diff[x-1:x+2,y-1:y+2]>=sig_fac_2) # cluster pixel above threshold
                
                if len(scp[0]) == 0:     #no other than the one in the middle
                    category = 0

                
                elif len(scp[0]) == 1:    #one other, so 2 in total
                    if len(np.where(scp[0] != 1)[0]) == 1 and len(np.where(scp[1] != 1)[0]) == 1: #discard if it is in a corner
                        category = 2
                        discard = True
                    else:
                        category = 1   #add all the stuff to the list...
                        tmp_event_sum += image[x-1 + scp[0][0]][y-1 + scp[1][0]]
                        tmp_event_x += (x-1 + scp[0][0])*image[x-1 + scp[0][0]][y-1 + scp[1][0]]
                        tmp_event_y += (y-1 + scp[1][0])*image[x-1 + scp[0][0]][y-1 + scp[1][0]]
                        tmp_event_n += 1
                        px_list_int += [image[x-1 + scp[0][0]][y-1 + scp[1][0]]]
                        px_list_pos += [(x-1 + scp[0][0],y-1 + scp[1][0])]
                        cluster_nr += 1
                        cluster_image[x-1 + scp[0][0]][y-1 + scp[1][0]] = cluster_nr
                        if scp[0][0] == 2:   #make sure no pixels in the extended cluster
                            if len(np.where(sigma_diff[x+2,y-1:y+2]>=sig_fac_2)[0]) > 0:   
                                category = 2
                                discard = True
                        if scp[0][0] == 0:
                            if len(np.where(sigma_diff[x-2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 2
                                discard = True
                        if scp[1][0] == 0:                      
                            if len(np.where(sigma_diff[x-1:x+2,y-2]>=sig_fac_2)[0]) > 0:
                                category = 2
                                discard = True
                        if scp[1][0] == 2:
                            if len(np.where(sigma_diff[x-1:x+2,y+2]>=sig_fac_2)[0]) > 0: 
                                category = 2
                                discard = True


                
                elif len(scp[0]) == 2:  #for three pixels in total...
                    if len(np.where(scp[0] != 1)[0]) == 1 and len(np.where(scp[1] != 1)[0]) == 1:           # for both above threshold pixels there is exactly one coordinate not in the middle (!= 1)
                        category = 3  #making sure the 3 pixel events having this rectangular shape
                        for i in range(len(scp[0])): #add to the list
                            tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                            tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                            tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                            tmp_event_n += 1
                            px_list_int += [image[x-1 + scp[0][i]][y-1+scp[1][i]]]
                            px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                            cluster_nr += 1
                            cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr
                        if len(np.where(scp[0] == 2)[0]) == 1:    #check extended cluster
                            if len(np.where(sigma_diff[x+2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 4
                                discard = True
                        if len(np.where(scp[0] == 0)[0]) == 1:
                            if len(np.where(sigma_diff[x-2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 4
                                discard = True
                        if len(np.where(scp[1] == 0)[0]) == 1:
                            if len(np.where(sigma_diff[x-1:x+2,y-2]>=sig_fac_2)[0]) > 0:
                                category = 4
                                discard = True
                        if len(np.where(scp[1] == 2)[0]) == 1:
                            if len(np.where(sigma_diff[x-1:x+2,y+2]>=sig_fac_2)[0]) > 0:
                                category = 4
                                discard = True
                            
                    else:
                        category = 4   #discard every 3 pixel event not fulfilling condition above
                        discard = True
                        
                elif len(scp[0]) == 3: #now for 4
                    if np.sum(scp[0]) + np.sum(scp[1]) == 10 or np.sum(scp[0]) + np.sum(scp[1]) == 2  or np.sum(scp[0]) + np.sum(scp[1]) == 6:
                        if len(np.where(scp[0] == 0)[0]) == 2 and len(np.where(scp[1] == 0)[0]) == 2:
                            category = 5 #"Four" #to make sure events have a square shape
                            for i in range(len(scp[0])): #add stuff...
                                tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_n += 1
                                px_list_int += [image[x-1+scp[0][i]][y-1+scp[1][i]]]
                                px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                                cluster_nr += 1
                                cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr
                            if len(np.where(sigma_diff[x-1:x+2,y-2]>=sig_fac_2)[0]) + len(np.where(sigma_diff[x-2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 6# "Four+"  #check extended cluster
                                discard = True
                            if sigma_diff[x-2,y-2]>=sig_fac_2: #pixel in the corner
                                category = 6# "Four+"
                                discard = True
                        elif len(np.where(scp[0] == 0)[0]) == 2 and len(np.where(scp[1] == 2)[0]) == 2:
                            category = 5# "Four"
                            for i in range(len(scp[0])):
                                tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_n += 1
                                px_list_int += [image[x-1 + scp[0][i]][y-1 + scp[1][i]]]
                                px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                                cluster_nr += 1
                                cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr
                            if len(np.where(sigma_diff[x-1:x+2,y+2]>=sig_fac_2)[0]) + len(np.where(sigma_diff[x-2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 6# "Four+"
                                discard = True
                            if sigma_diff[x-2,y+2]>=sig_fac_2:
                                category = 6# "Four+"
                                discard = True
                        elif len(np.where(scp[0] == 2)[0]) == 2 and len(np.where(scp[1] == 0)[0]) == 2:
                            category = 5# "Four"
                            for i in range(len(scp[0])):
                                tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_n += 1
                                px_list_int += [image[x-1 + scp[0][i]][y-1 + scp[1][i]]]
                                px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                                cluster_nr += 1
                                cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr
                            if len(np.where(sigma_diff[x-1:x+2,y-2]>=sig_fac_2)[0]) + len(np.where(sigma_diff[x+2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 6# "Four+"
                                discard = True
                            if sigma_diff[x+2,y-2]>=sig_fac_2:
                                category = 6# "Four+"
                                discard = True
                        elif len(np.where(scp[0] == 2)[0]) == 2 and len(np.where(scp[1] == 2)[0]) == 2:
                            category = 5# "Four"
                            for i in range(len(scp[0])):
                                tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                                tmp_event_n += 1
                                px_list_int += [image[x-1 + scp[0][i]][y-1 + scp[1][i]]]
                                px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                                cluster_nr += 1
                                cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr
                            if len(np.where(sigma_diff[x-1:x+2,y+2]>=sig_fac_2)[0]) + len(np.where(sigma_diff[x+2,y-1:y+2]>=sig_fac_2)[0]) > 0:
                                category = 6# "Four+"
                                discard = True
                            if sigma_diff[x+2,y+2]>=sig_fac_2:
                                category = 6# "Four+"
                                discard = True
                    else:
                        category = 6# "Four+" #get rid of everything else
                        discard = True
                     
                elif len(scp[0]) == 8: #if all pixels in the 3x3 cluster are above sig_fac_2
                    category = 7# "All"
                    for i in range(len(scp[0])): #add...
                        tmp_event_sum += image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                        tmp_event_x += (x-1 + scp[0][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                        tmp_event_y += (y-1 + scp[1][i])*image[x-1 + scp[0][i]][y-1 + scp[1][i]]
                        tmp_event_n += 1
                        px_list_int += [image[x-1 + scp[0][i]][y-1 + scp[1][i]]]
                        px_list_pos += [(x-1 + scp[0][i],y-1 + scp[1][i])]
                        cluster_nr += 1
                        cluster_image[x-1 + scp[0][i]][y-1 + scp[1][i]] = cluster_nr

                else:
                    category = 8# "Whatever" #discard if its anything else
                    discard = True
                
                icx,icy = np.where(sigma_diff[x-1:x+2,y-1:y+2]>=sig_fac_2)
                for x,y in zip(icx,icy):
                    sigma_diff[x,y] = 0
                                
                count_list[category] += 1
                if discard == False:
                    px_x.append(round(1.*tmp_event_x/(tmp_event_sum),2))
                    px_y.append(round(1.*tmp_event_y/(tmp_event_sum),2))
                    px_I.append(tmp_event_sum)
                    px_n.append(tmp_event_n)
                    px_I_rel.append(px_list_int)
                    px_Pos_rel.append(px_list_pos)
                    if category == 0:#categories.index('Single'):
                        cluster_nr += 1
                    cluster_image[x][y] = cluster_nr

    return px_x, px_y, px_I, px_n, px_I_rel, px_Pos_rel, cluster_image


######################
#####   QGMF   #######
def qgmf(image, sig_fac_1, squ_a, sigma, defect_map=None):
    """
    Evaluate single photon events on a 2d array (e.g. ccd, cmos) using a linear
    fit after taking the log of the image (in principle gaussian fit)

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    squ_a : int
        defining the edge length of the box of the subarray used for fitting in
    sigma : float
        expected standard deviation of the noise in the image
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.

        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)
    tmp_event_n : int
        number of pixels the event cluster consists of
    px_list_int : list of floats
        intensities of each pixel belonging to the cluster
    px_list_pos : list of tuples (float,float)
        (x,y) position of each pixel belonging to the cluster

    Author
    ------
    Jonas Baumann, acknowledgements: JB likes to thank the unknown professor at the ML conference in Munich for providing the idea
    """
    
    if not hasattr(sigma,"__len__"):    
        sigma_image = np.zeros(np.shape(image))
        sigma_image.fill(sigma)
    else:   
        sigma_image = np.array(sigma)
        sigma = np.mean(sigma_image)
    sigma_image = np.array(sigma_image,dtype='float')
    image = np.array(image)
    [x_max, y_max] = np.shape(image)
    [significant_x_values, significant_y_values] = get_significant_pixels(image, sigma_image, sig_fac_1, defect_map)
    sigma_diff = image/np.array(sigma_image,dtype='float')              # to ensure no rounding
    with np.errstate(invalid='ignore', divide='ignore'):
        image_log = np.log(image)
    image_log = np.nan_to_num(image_log, nan=0.0, posinf=0, neginf=-127)
    
    photon_events = qgmf_jit(image, image_log, sigma_diff, squ_a, sigma_image, x_max, y_max, significant_x_values, significant_y_values)
    return list(map(list, zip(*photon_events[:-3]))), photon_events[-3], photon_events[-2], photon_events[-1]


@jit(nopython=True, cache=False)
def simp2param(simp, imx, imy):
    eps = 1e-42
    sigma_x = simp[3]+eps
    sigma_y = sigma_x*simp[4]
    A = simp[2]/(2*np.pi*sigma_x*sigma_y)
    x0 = simp[0]+(0.5*imx-0.5)              # ToDo: Double check if adding -0.5 is correct (not critical, because it is not used, yet)
    y0 = simp[1]+(0.5*imy-0.5)              # ToDo: Double check if adding -0.5 is correct (not critical, because it is not used, yet)
    pgt = np.zeros(5)
    pgt[0] = np.log(A)-x0**2/(2*sigma_x**2)-y0**2/(2*sigma_y**2)
    pgt[1] = x0/sigma_x**2
    pgt[3] = y0/sigma_y**2
    pgt[2] = -1/(2*sigma_x**2)
    pgt[4] = -1/(2*sigma_y**2)
    return pgt


@jit(nopython=True, cache=False)
def param2simp(param, imx, imy):
    eps = 1e-42
    sigma_x = np.sqrt(np.abs(1/(2*param[2]+eps)))
    sigma_y = np.sqrt(np.abs(1/(2*param[4]+eps)))
    x0 = param[1]*sigma_x**2 - (0.5*imx-0.5)        # if e.g. x = 0,1,2 in case of squ_a=3 then middle is 1 --> squ-a/2-0.5
    y0 = param[3]*sigma_y**2 - (0.5*imy-0.5)
    V = 2*np.pi*sigma_x*sigma_y * np.exp(param[0]+(x0+(0.5*imx-0.5))**2/(2*sigma_x**2+eps)+(y0+(0.5*imy-0.5))**2/(2*sigma_y**2+eps))
    return np.array([x0,y0,V,sigma_x,sigma_y])


@jit(nopython=True, cache=False)
def fit(M, b, w):
    Mw = M * w[:, np.newaxis]
    bw = b * w
    x = np.linalg.lstsq(Mw, bw)[0]
    im_r = np.sum(M*x, axis=1)
    return x, im_r


#@jit(nopython=True, cache=False)
def qgmf_jit(image, image_log, sigma_diff, squ_a, sigma_image, x_max, y_max, significant_x_values, significant_y_values):
    imsize = np.shape(image)
    squ_ah = int(squ_a/2.0)

    residuum_image = np.zeros(np.shape(image))
    recon_image = np.zeros((x_max,y_max))
    
#    # construct M
    xlen, ylen = squ_a, squ_a                   # just if in the future rectangular sub images will be used, if so ... double check xlen ylen definition!
    M = np.zeros((xlen*ylen, 5))
    #x = np.tile(np.arange(xlen, dtype=np.float32), ylen)
    x = np.repeat(np.arange(ylen, dtype=np.float32), xlen).reshape(ylen, xlen).T.ravel()
    x2 = x**2
    y = np.repeat(np.arange(xlen, dtype=np.float32), ylen)
    y2 = y**2
    M[:,0] = 1.
    M[:,1] = x
    M[:,2] = x2
    M[:,3] = y
    M[:,4] = y2

    pe_x = [1.]
    pe_y = [1.]
    pe_I = [1.]
    pe_sig_x = [1.]
    pe_sig_y = [1.]
    pe_err = [[1.]*5]
    pe_mse = [1.]
    fails = 0
    
    del pe_x[:]
    del pe_y[:]
    del pe_I[:]
    del pe_sig_x[:]
    del pe_sig_y[:]
    del pe_err[:]
    del pe_mse[:]


    for i in range(0,len(significant_x_values)):
        x = significant_x_values[i]
        y = significant_y_values[i]    

        if contains_box(x,y,np.shape(image)[0],np.shape(image)[1],squ_a+2):
                
            s = image[x-int(squ_a/2):x+int(np.ceil(squ_a/2)),y-int(squ_a/2):y+int(np.ceil(squ_a/2))]            # maybe take residuum image for new fits
            s_log = image_log[x-int(squ_a/2):x+int(np.ceil(squ_a/2)),y-int(squ_a/2):y+int(np.ceil(squ_a/2))]
            if image[x][y] == np.max(s):
                sigma_diff[x][y] = 0
                
                b = np.ravel(s_log)
                w = np.ravel(s)
                w[w<0] = 1e-42
                param, im_r = fit(M, b, w)
                im_r = im_r.reshape((xlen, ylen))
                x0, y0, volume, sig_x, sig_y = param2simp(param, squ_a, squ_a)
                if not np.isfinite(volume) or x0<-squ_ah or y0<-squ_ah:
                    fails += 1
                else:
                    pos_x, pos_y = x-squ_ah+x0, y-squ_ah+y0                                     ### GANZ EVTL NOCH + 0.5
                    err_list = [0.]*5                                                                # not included, yet
                    residuum_image[x-int(squ_a/2):x+int(np.ceil(squ_a/2)),y-int(squ_a/2):y+int(np.ceil(squ_a/2))] = s-im_r               # set the nxn area in the residuum image to the residuum
                    recon_image[x-int(squ_a/2):x+int(np.ceil(squ_a/2)),y-int(squ_a/2):y+int(np.ceil(squ_a/2))] = im_r                     # set the nxn area in the residuum image to the residuum
                    mean_sq_err = np.sum((s-im_r)**2)/squ_a**2
                    pe_x.append(round(pos_x,2))
                    pe_y.append(round(pos_y,2))
                    pe_I.append(volume)
                    pe_sig_x.append(sig_x)
                    pe_sig_y.append(sig_y)
                    pe_err.append(err_list)
                    pe_mse.append(mean_sq_err)
                
    return pe_x, pe_y, pe_I, pe_sig_x, pe_sig_y, pe_err, pe_mse, recon_image, residuum_image, fails




####################
#####   NN   #######
def NN(image, sig_fac_1, sigma_image, network, image_scaling=1., image_splits=4, defect_map=None):
    """
    Evaluate single photon events on a 2d array (e.g. ccd, cmos) using a pretrained
    Neural Network. WARNING: Pytorch needs to be installed!

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    network : str or pytorch_lightning.LightningModule
        name of network to be used or net itself
    ret_cluster_image : bool, optional
        if True, an image (2d-array) containing the cluster assignment and
        numbering is returned additionally to the usual output.

        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    intensity : float
        total event intensity (proportional to energy of photon)

    Author
    ------
    Jonas Baumann
    """
    print('image max / ADU =', np.max(image))
    image = np.array(image)*image_scaling
    
    [x_max, y_max] = np.shape(image)
    process = lambda inp: inp.clone().detach().cpu().numpy()
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    if isinstance(network, str):
        spe_dir = os.path.abspath(inspect.getfile(inspect.currentframe())).rsplit(os.sep,1)[0]
        network = torch.jit.load(os.path.join(spe_dir, 'nn_models', network+'.pt'))

    network.to(DEVICE)
    if image_splits < 2:
        im_batch = torch.tensor(image, dtype=torch.get_default_dtype()).to(DEVICE)
        pred = process(network(im_batch[None,:,:]).squeeze())
    else:
        im_batch = torch.tensor(np.array(split_image(image, image_splits)), dtype=torch.get_default_dtype()).to(DEVICE)
        pred_sub = np.zeros(im_batch.size())
        for i,im in enumerate(im_batch):
            pred_sub[i,:,:] = process(network(im[None,:,:]).squeeze())
        pred = stitch_images(pred_sub, image_splits)

    pred = pred/image_scaling
    [significant_x_values, significant_y_values] = get_significant_pixels(pred, sigma_image, sig_fac_1, defect_map)
    z = pred[significant_x_values, significant_y_values]
    return list(np.array([significant_x_values, significant_y_values, z]).T), pred

    
def split_image(image, n):
    # Determine the number of subimages per dimension
    per_dim = int(np.sqrt(n))

    if per_dim ** 2 != n:
        raise ValueError("n must be a perfect square")
    if image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")

    subimage_size = image.shape[0] // per_dim
    subimages = []

    for i in range(per_dim):
        for j in range(per_dim):
            subimage = image[i*subimage_size:(i+1)*subimage_size, j*subimage_size:(j+1)*subimage_size]
            subimages.append(subimage)

    return subimages


# Function to stitch subimages back into the original image
def stitch_images(subimages, n):
    per_dim = int(np.sqrt(n))
    image_size = per_dim * subimages[0].shape[0]  # Assuming all subimages are the same size
    image = np.zeros((image_size, image_size), dtype=subimages[0].dtype)

    for idx, subimage in enumerate(subimages):
        i, j = divmod(idx, per_dim)
        subimage_size = subimage.shape[0]
        image[i*subimage_size:(i+1)*subimage_size, j*subimage_size:(j+1)*subimage_size] = subimage

    return image