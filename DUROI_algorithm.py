# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:20:00 2020

@author: lykha
"""

import random
from scipy.stats import uniform, norm
import numpy as np
from DUROI_Preprocess import *


def search_nearest_from_L(r, i, L, k):
    R = len(L)
    K = range(i-k, i+ k + 1 )
    K = [ get_circular_index(i_, R)  for i_ in K ]
    return K

def smooth_Fct(roi, r, i, L, Fct_L, a, voxelsize, k, gradct):
    K = search_nearest_from_L(r, i, L, k)
    K_after = np.array(K) % len(L)
    K_after = K_after[K_after > 0]
    Fct_r = find_Fct(roi, r, i, a, voxelsize, gradct)
    Fct_k = np.array([Fct_L[int(k)] for k in K_after])
    
    Fct_x = 1/(2*k+1)*(Fct_r[0] + sum(Fct_k[..., 0]))
    Fct_y = 1/(2*k+1)*(Fct_r[1] + sum(Fct_k[..., 1]))
    Fct_r = np.array([float(Fct_x), float(Fct_y)])
    return Fct_r

def find_Fsd(SD, seed): 
    random.seed(seed)
    pRand_k = uniform.rvs(0, 1, size = 1)
    Fsd_x=norm.ppf(pRand_k, loc=0, scale=SD[0])
    random.seed(seed+1)
    pRand_k = uniform.rvs(0, 1, size = 1)
    Fsd_y=norm.ppf(pRand_k, loc=0, scale=SD[1])
    random.seed(seed+2)
    pRand_k = uniform.rvs(0, 1, size = 1)
    Fsd_z=norm.ppf(pRand_k, loc=0, scale=SD[2])
    Fsd_r = np.array([float(Fsd_x), float(Fsd_y), float(Fsd_z)])
    return Fsd_r

def find_Fct(roi, r, i, a, voxelsize, gradct):
    gradct_x = gradct[0][r, i]
    gradct_y = gradct[1][r, i]
    #gradct_z = gradct[2]
    Fct_x = a/(np.abs(gradct_x)+a)*np.copysign(1,  gradct_x)
    Fct_y = a/(np.abs(gradct_y)+a)*np.copysign(1,  gradct_y)
    Fct_r = np.array([float(Fct_x), float(Fct_y)])
    return Fct_r

def find_D(Fsd_r, Fct_r):
    Fct_x =  Fct_r[0]
    Fct_y =  Fct_r[1]
    Fsd_x =  Fsd_r[0]
    Fsd_y =  Fsd_r[1]
    D_x = Fsd_x*Fct_x
    D_y = Fsd_y*Fct_y
    D_r = [float(D_x), float(D_y)]
    return D_r

def r_to_xyz(F):
    F_x =  F[0]
    F_y =  F[1]
    #F_z =  F[2]
    return F_x, F_y

def assd_Sobel(slices, target_label, voxelsize, a, SD, circles, seed, k, w, images, labels, organ_i, smooth=True):
    mask = np.where(target_label!=0,4,0)
    surface, interior = make_surface_contour(mask)
    roi=slices*mask
    row_size = roi.shape[0]
    col_size = roi.shape[1]
    mat = np.ndarray([row_size, col_size],dtype=np.float64)
    dx = np.zeros((512, 512))
    dy = np.zeros((512, 512))
    #dz = np.zeros((512, 512))
    surface_cord = np.argwhere(surface != 0)
    start = random.choice(surface_cord.tolist())
    roi_z = find_roi_slices(images, labels, organ_i)
    j = 0
    gradct = find_Sobel_gradct(roi)
    L, Fct_L = order_voxel_list(start, surface_cord, roi, a, voxelsize, gradct)
    i0 = find_i0(images, labels, roi_z, surface_cord, L, voxelsize, roi)   
    t = find_tg(L, w, i0)
    
    for r in range(0,row_size -1):
        for i in range(0,col_size-1): 
            i = int(i)
            r = int(r)
            if  surface[r, i] != 0:
                Fsd_r = find_Fsd(SD, seed)
                #pq, L, Fct_L = find_pd(j, start, surface_cord, circles)
                if (smooth):
                    Fct_r = smooth_Fct(slices, r, i, L, Fct_L, a, voxelsize, k, gradct)
                else:
                    Fct_r = find_Fct(slices, r, i, a, voxelsize, gradct)
                    
                D_r = find_D(Fsd_r, Fct_r)
               
                D_x, D_y = r_to_xyz(D_r)
                
                dx[r, i] = D_x*t[j]
                dy[r, i] = D_y*t[j] 
                
                 
            elif interior[r, i] != 0: 
                pos_r, neg_r = nearest_neighbor_search(surface[r], i)
                dx[r, i] = 0.00000000000001 #D_x
                dy[r, i] = 0.00000000000001 #D_y
            
            
    return dx, dy, mask, t, L, roi_z


