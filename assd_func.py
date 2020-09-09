import numpy as np 
import pandas as pd 
import os
import scipy.ndimage
import math
import random
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from scipy.stats import uniform,norm
import matplotlib.pyplot as plt
import scipy.stats as sct
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from skimage import morphology
from skimage.draw import polygon
import cv2

def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        if i == 3:
            contour = {}
            contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber ##
            contour['name'] = structure.StructureSetROISequence[i].ROIName
            contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append(contour)
    return contours

def get_mask(contours, slices, image):
    z = [round(s.ImagePositionPatient[2],1) for s in slices] ##
    #print(z)
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]

    label = np.zeros_like(image, dtype=np.uint8)
    for con in contours:
        num = int(con['number'])
        
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3)) #triplets describing points of contour
            #assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
  
            #print(np.around(nodes[0, 2], 1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = num
            #con['color'] = [0, 0, 255] 
            colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return label, colors

def make_surface_contour(mask):
    interior = morphology.erosion(mask,np.ones([3,3])) # one last dilation 
    contour = np.where(interior==0, 1, 0)
    surface = contour*mask
    return surface, interior

def search(vector, r, i, pos_neighbor, neg_neighbor, pos_current_distance, neg_current_distance):
    if vector[i] != 0:   
        if i-r > 0:
            if i-r < min(pos_current_distance):
                pos_current_distance.append(i-r)
                #pos_neighbor = vector[i]
                pos_neighbor = i
            
        elif i-r < 0:
            if abs(i-r) < min(neg_current_distance):
                neg_current_distance.append(abs(i-r))
                #neg_neighbor = vector[i]
                neg_neighbor = i
    return pos_neighbor, neg_neighbor

def nearest_neighbor_search(vector, r):
    pos_neighbor = 0
    neg_neighbor = 0
    pos_current_distance = [len(vector)]
    neg_current_distance = [len(vector)]
    for i in range(0, len(vector)-1):
        pos_neighbor, neg_neighbor = search(vector, r, i, pos_neighbor, neg_neighbor, pos_current_distance, neg_current_distance)
    return pos_neighbor, neg_neighbor

def find_min_dist(start, surface_cord):
    dist_list = []
    for i in surface_cord:
        x_j = i[0]
        y_j = i[1]
        dist = ((x_j - start[0])**2)**0.5 + ((y_j - start[1])**2)**0.5
        dist_list.append(dist)
        min_dist = min(dist_list)
    return min_dist
    
def find_next_voxel(start, surface_cord):
    min_dist = find_min_dist(start, surface_cord)
    L = []
    for i in surface_cord:
        x_j = i[0]
        y_j = i[1]
        dist = ((x_j - start[0])**2)**0.5 +  ((y_j - start[1])**2)**0.5
        if dist == min_dist:
            next_voxel = [x_j, y_j]
    return next_voxel
    
def order_voxel_list(start, surface_cord, roi, a, voxelsize):
    surface_cord = surface_cord.tolist()
    L = [start]
    index = surface_cord.index(start)
    del surface_cord[index]
    for i in range(len(surface_cord)):
        try:
            next_voxel = find_next_voxel(start, surface_cord)
        except:
            for voxel in surface_cord:
                if i not in L:
                    next_voxel = voxel
                    
        L.append(next_voxel)
        start = next_voxel
        index = surface_cord.index(start)
        del surface_cord[index]
        
    Fct_L = []
    for l in L:
        Fct_L.append(list(find_Fct(roi, l[0], l[1], a, voxelsize)))
    Fct_L = np.array(Fct_L)
    
    return L, Fct_L

def find_gradct(r, i, voxelsize, roi):
    gradct = (roi[r+1,i]-roi[r-1, i])/(2*voxelsize)
    return gradct

def find_i0(L, voxelsize, roi, ismax):
    gradct_list = []
    L = np.array(L)
    for i in range(L.shape[0]):
        gradct = find_gradct(L[..., 0][i], L[..., 1][i], voxelsize, roi)
        grad_dist = (gradct[0]**2) + (gradct[1]**2) 
        gradct_list.append(grad_dist)
    if (ismax):
        i0 = gradct_list.index(max(gradct_list))
    else:
        i0 = gradct_list.index(min(gradct_list))
    return i0

def get_circular_index(i, R):
    if i < 0:
        r = np.ceil(-i/R)*R + i
    else:
        r = i % R
    return r

def calc_circular_dist(i, i0, R):
    circular_dist = min([(i-i0)**2, (i-i0 - R)**2,  (i-i0 + R)**2])
    return circular_dist

def calc_tg(r, i, start, surface_cord, a, voxelsize, roi, w, ismax):
    L, Fct_L = order_voxel_list(start, surface_cord, roi, a, voxelsize)
    R = len(L)
    i0 = find_i0(L, voxelsize, roi, ismax)
    w0 = R/w
    i = L.index([r, i])
    #i = i - i0
    #i = get_circular_index(i, R)
    r = np.arange(i)
    circular_dist = calc_circular_dist(i, i0, R)
    t = 1/(np.sqrt(2*np.pi)*w0)*np.exp(-(circular_dist)/(2*w0**2)) 
    return t

def find_tg(L, start, surface_cord, a, voxelsize, roi, w, ismax):
    tg_list = []
    for i in range(len(L)):
        tg = calc_tg(L[i][0], L[i][1], start, surface_cord, a, voxelsize, roi, w, ismax)
        tg_list.append(tg)
    t = np.array(tg_list) - np.mean(tg_list) 
    return t*1000

def search_nearest_from_L(r, i, L, k):
    i = L.index([r, i])
    if i-k >= 0 and i+k+1 <= len(L):
        K = list(range(i-k, i+k+1))
        #K_list = L[i-k : i+k+1]
    elif i-k < 0 and i+k+1 <= len(L):
        K = list(range((i + len(L) - k), len(L))) + list(range(0, i+k+1))    #K_list = L[0 : i+k+1]
    elif i-k >= 0 and i+k+1 > len(L):
        K = list(range(i-k, len(L))) + list(range(0, (i + k - len(L)) + 1))
    elif i-k <0 and i+k+1 > len(L):
        K = range(0, len(L))
        #K_list = L[i-1, len(L)]
    return K

def search_nearest_from_L(r, i, L, k):
    R = len(L)
    K = range(i-k, i+ k + 1 )
    K = [ get_circular_index(i_, R)  for i_ in K ]
    return K

def smooth_Fct(roi, r, i, L, Fct_L, a, voxelsize, k):
    K = search_nearest_from_L(r, i, L, k)
    K_after = np.array(K) % len(L)
    K_after = K_after[K_after > 0]
    Fct_r = find_Fct(roi, r, i, a, voxelsize)
    Fct_k = np.array([Fct_L[int(k)] for k in K_after])
    
    Fct_x = 1/(2*k+1)*(Fct_r[0] + sum(Fct_k[..., 0]))
    Fct_y = 1/(2*k+1)*(Fct_r[1] + sum(Fct_k[..., 1]))
    Fct_z = 1/(2*k+1)*(Fct_r[2] + sum(Fct_k[..., 2]))
    Fct_r = np.array([float(Fct_x), float(Fct_y), float(Fct_z)])
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

def find_Fct(roi, r, i, a, voxelsize):
    gradct = find_gradct(r, i, voxelsize, roi)
    gradct_x = gradct[0]
    gradct_y = gradct[1]
    gradct_z = gradct[2]
    Fct_x = a/(np.abs(gradct_x)+a)*np.copysign(1,  gradct_x)
    Fct_y = a/(np.abs(gradct_y)+a)*np.copysign(1,  gradct_y)
    Fct_z = a/(np.abs(gradct_z)+a)*np.copysign(1,  gradct_z)
    Fct_r = np.array([float(Fct_x), float(Fct_y), float(Fct_z)])
    return Fct_r
        
def find_Fct_interior_voxel(neg_Fct_r, pos_Fct_r, pos_r, neg_r, r, voxelsize):
    Fct_x = neg_Fct_r[0] + (pos_Fct_r[0] - neg_Fct_r[0])/((pos_r - neg_r)*voxelsize[0])*((r - neg_r)*voxelsize[0])
    Fct_y = neg_Fct_r[1] + (pos_Fct_r[1] - neg_Fct_r[1])/((pos_r - neg_r)*voxelsize[1])*((r - neg_r)*voxelsize[1])
    Fct_z = neg_Fct_r[2] + (pos_Fct_r[2] - neg_Fct_r[2])/((pos_r - neg_r)*voxelsize[2])*((r - neg_r)*voxelsize[2])
    Fct_r = np.array([float(Fct_x), float(Fct_y), float(Fct_z)])
    return Fct_r

def find_D(Fsd_r, Fct_r):
    Fct_x =  Fct_r[0]
    Fct_y =  Fct_r[1]
    Fct_z =  Fct_r[2]
    Fsd_x =  Fsd_r[0]
    Fsd_y =  Fsd_r[1]
    Fsd_z =  Fsd_r[2]
    D_x = Fsd_x*Fct_x
    D_y = Fsd_y*Fct_y
    D_z = Fsd_z*Fct_z
    D_r = [float(D_x), float(D_y), float(D_z)]
    return D_r

def r_to_xyz(F):
    F_x =  F[0]
    F_y =  F[1]
    F_z =  F[2]
    return F_x, F_y, F_z

def assd(slices, target_label, voxelsize, a, SD, circles, seed, k, w, smooth=True, blur=False, ismax=False):
    mask = np.where(target_label!=0,4,0)
    surface, interior = make_surface_contour(mask)
    roi=slices*mask
    row_size = roi.shape[0]
    col_size = roi.shape[1]
    mat = np.ndarray([row_size, col_size],dtype=np.float64)
    dx = np.zeros((512, 512))
    dy = np.zeros((512, 512))
    dz = np.zeros((512, 512))
    surface_cord = np.argwhere(surface != 0)
    start = random.choice(surface_cord.tolist())
    j = 0
    L, Fct_L = order_voxel_list(start, surface_cord, roi, a, voxelsize)
    t = find_tg(L, start, surface_cord, a, voxelsize, roi, w, ismax)
    if (blur):
        slices = cv2.GaussianBlur(slices,(25,25),0) 
    
    for r in range(0,row_size -1):
        for i in range(0,col_size-1): 
            i = int(i)
            r = int(r)
            if  surface[r, i] != 0:
                Fsd_r = find_Fsd(SD, seed)
                #pq, L, Fct_L = find_pd(j, start, surface_cord, circles)
                if (smooth):
                    Fct_r = smooth_Fct(slices, r, i, L, Fct_L, a, voxelsize, k)
                else:
                    Fct_r = find_Fct(slices, r, i, a, voxelsize)
                D_r = find_D(Fsd_r, Fct_r)
                D_x, D_y, D_z = r_to_xyz(D_r)
                
                if t[j] == 0:
                    print("yes")
                    t[j] = 0.000001
                    
                else:
                    t[j] = t[j]
                
                dx[r, i] = D_x*t[j]
                dy[r, i] = D_y*t[j] 
                
                 
            elif interior[r, i] != 0: 
                pos_r, neg_r = nearest_neighbor_search(surface[r], i)
                dx[r, i] = 0.00000000000001 #D_x
                dy[r, i] = 0.00000000000001 #D_y
            
            
    return dx, dy, mask, t, L

def plotting_assd(dx, dy, mask, target_img, quiver=False, plot=True, display=False):
    roi_cord = np.argwhere(mask != 0)
    x = []
    y = []
    
    for i in roi_cord.tolist():
        x.append(i[0])
        y.append(i[1])
    x = np.array(x)
    y = np.array(y)
   
    u = []
    v = []
    w = []
    u_cord = np.argwhere(dx != 0)
    v_cord = np.argwhere(dy != 0)
    for i in u_cord.tolist():
        u.append(dx[i[0], i[1]])
    for i in v_cord.tolist():
        v.append(dy[i[0], i[1]])
    
    u = np.array(u)
    v = np.array(v)
          
    if (quiver): #False by default
        fig,ax = plt.subplots()
        ax.quiver(x, y, u, v)
        plt.show()
    
    DU_mask = np.zeros((target_img.shape[0],target_img.shape[1]))
    x_new = x + u
    y_new = y + v
    
    for i in range(len(x_new)-1):
        DU_mask[int(round(x_new[i], 0)), int(round(y_new[i], 0))] = 1
        #DU_mask[int(x[i]), int(y[i])] = 1
        
    du = make_mask(DU_mask, display)
    if (plot):
        fig,ax = plt.subplots(1,1,figsize=[12,12])
        plt.imshow(target_img)
        ax.contour(mask, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="blue")
        ax.contour(du, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="red")
        ax.set_xlim(384, 128)
        ax.set_ylim(384, 128)
        plt.show()
        
    return du

def make_mask(img, display):
    threshold = np.mean(img)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([7,7]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    #labels = measure.label(dilation)
    blur = cv2.GaussianBlur(dilation,(25,25),0)
    blur = cv2.GaussianBlur(blur,(25,25),0)
    #blur = cv2.GaussianBlur(blur,(25,25),0)

    return blur




