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
from scipy import ndimage
import cv2 as cv
import numpy as np
import matplotlib.patches as mpatches

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


def search_nearest_from_L(r, i, L, k):
    R = len(L)
    K = range(i-k, i+ k + 1 )
    K = [ get_circular_index(i_, R)  for i_ in K ]
    return K

def smooth_Fct(roi, r, i, L, Fct_L, a, voxelsize, k, gradct):
    index = L.index([r, i])
    K = search_nearest_from_L(r, i, L, k)
    K_after = np.array(K) % len(L)
    K_after = K_after[K_after > 0]
    Fct_r =   Fct_L[index] #find_Fct(roi, r, i, a, voxelsize, gradct)
    Fct_k = np.array([Fct_L[int(k)] for k in K_after])
    
    Fct_x = 1/(2*k+1)*(Fct_r[0] + sum(Fct_k[..., 0]))
    Fct_y = 1/(2*k+1)*(Fct_r[1] + sum(Fct_k[..., 1]))
    Fct_r = np.array([float(Fct_x), float(Fct_y)])
    return Fct_r

def find_Fsd(SD, seed): 
    random.seed(seed)
    pRand_k = random.random()
    Fsd_x=norm.ppf(pRand_k, loc=0, scale=SD[0])
    random.seed(seed+1)
    pRand_k = random.random()
    Fsd_y=norm.ppf(pRand_k, loc=0, scale=SD[1])
    random.seed(seed+2)
    pRand_k = random.random()
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
    #mat = np.ndarray([row_size, col_size],dtype=np.float64)
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
                    print(Fct_r)
                else:
                    index = L.index([r, i])
                    Fct_r = Fct_L[index] #find_Fct(slices, r, i, a, voxelsize, gradct)
                    
                D_r = find_D(Fsd_r, Fct_r)
               
                D_x, D_y = r_to_xyz(D_r)
                
                dx[r, i] = D_x*t[j]
                dy[r, i] = D_y*t[j] 
                
                 
            elif interior[r, i] != 0: 
                pos_r, neg_r = nearest_neighbor_search(surface[r], i)
                dx[r, i] = 0.00000000000001 #D_x
                dy[r, i] = 0.00000000000001 #D_y
            
            
    return dx, dy, mask, t, L, roi_z


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
    blur = cv.GaussianBlur(dilation,(25,25),0)
    #blur = cv2.GaussianBlur(blur,(25,25),0)
    #blur = cv2.GaussianBlur(blur,(25,25),0)
    final_du = np.where(blur < 0.5, 0, 4)
    return final_du

