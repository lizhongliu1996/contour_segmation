# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:09:15 2020

@author: lykha
"""
from skimage import morphology
from skimage.draw import polygon
import cv2 as cv
import numpy as np

#1 for bladder, 2 for rectum, 3 for prostate, 4 for Seminal vesicles, 6 for femoral head

def read_structure(structure, organ_id):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        if i == organ_id:
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
    #print(dist_list)
    if dist_list == []:
        min_dist = 0
    else:
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
    
def order_voxel_list(start, surface_cord, roi, a, voxelsize, gradct):
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
        Fct_L.append(list(find_Fct(roi, l[0], l[1], a, voxelsize, gradct)))
    Fct_L = np.array(Fct_L)
    
    return L, Fct_L

def find_Sobel_gradct(src):
    # Gaussion blur
    src = cv.GaussianBlur(src, (1, 1), 0)
    ddepth = cv.CV_16S
    gray = src #cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # New grad by Sobel derivatives
    scale = 1
    delta = 0
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    gradct = np.array([grad_x, grad_y])
    return gradct

def find_roi_slices(images, labels, i):
    roi_z = []
    for j in range(images.shape[0]):
        if True in np.unique(labels[..., i][j, ...] > 0):
            roi_z.append(j)
    return roi_z

def voxel_z_closest_list(images, labels, start, roi_z, surface_cord, voxelsize, roi):
    z_roi_voxel_list = []
    grad_x_list = []
    grad_y_list = []
    for z in roi_z:
        target_img_next = images[z, ...]
        target_label_next = labels[..., 1][z, ...]
        roi = target_img_next*target_label_next
        gradct_slice = find_Sobel_gradct(roi)
        mask = np.where(target_label_next !=0,4,0)
        surface_next, interior_next = make_surface_contour(mask)
        surface_cord_next = np.argwhere(surface_next != 0)
        try:
            next_voxel = find_next_voxel(start, surface_cord)
        except:
            for voxel in surface_cord:
                if i not in L:
                    next_voxel = voxel
        z_roi_voxel_list.append(next_voxel)
        gradct_x = gradct_slice[0][next_voxel[0], next_voxel[1]]
        gradct_y = gradct_slice[1][next_voxel[0], next_voxel[1]]
        grad_x_list.append(gradct_x)
        grad_y_list.append(gradct_y)
    avg_grad_x = sum(grad_x_list)/len(roi_z)
    avg_grad_y = sum(grad_y_list)/len(roi_z) 
    avg_grad = [avg_grad_x, avg_grad_y]
    return z_roi_voxel_list, avg_grad

def find_z_list(images, labels, roi_z, surface_cord, voxelsize, roi):
    z_surface_list = []
    avg_grad_surface_list = []
    for start in surface_cord:
        z_roi_voxel_list, avg_grad = voxel_z_closest_list(images, labels, start, roi_z, surface_cord, voxelsize, roi)
        z_surface_list.append(z_roi_voxel_list)
        avg_grad_surface_list.append(avg_grad)
    return z_surface_list, avg_grad_surface_list

def find_i0(images, labels, roi_z, surface_cord, L, voxelsize, roi):
    #L = np.array(L)
    z_list, avg_grad_surface_list = find_z_list(images, labels, roi_z, surface_cord, voxelsize, roi)
    grad_dist = list((np.array(avg_grad_surface_list)[..., 0]**2) + (np.array(avg_grad_surface_list)[..., 1]**2)) #magnitude
    i0 = grad_dist.index(min(grad_dist))
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

def calc_tg(L, i, w, i0):
    #L, Fct_L = order_voxel_list(start, surface_cord, roi, a, voxelsize, gradct)
    R = len(L)
    r = L[i][0]
    i = L[i][1]
    #i0 = find_i0(roi_z, surface_cord, L, voxelsize, roi, gradct)
    w0 = R/w
    i = L.index([r, i])
    #i = i - i0
    #i = get_circular_index(i, R)
    r = np.arange(i)
    circular_dist = calc_circular_dist(i, i0, R)
    t = 1/(np.sqrt(2*np.pi)*w0)*np.exp(-(circular_dist)/(2*w0**2)) 
    return t

def find_tg(L, w, i0):
    tg_list = []
    for i in range(len(L)):
        tg = calc_tg(L, i, w, i0)
        tg_list.append(tg)
    t = np.array(tg_list) - np.mean(tg_list) 
    return t*1000