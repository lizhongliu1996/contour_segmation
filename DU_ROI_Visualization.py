# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:27:38 2020

@author: lykha
"""
from DUROI_Algorithm import * 

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

def different_view_plot(frontal_slice, sagittal_slice):
    frontal_target_img = images[0:100, frontal_slice, 0:512]
    frontal_target_label = labels[..., 6][0:100, frontal_slice, 0:512]
    dx, dy, mask, t, L = af_Sobel.assd_Sobel(frontal_target_img, frontal_target_label, voxelsize, a, SD, circles, seed, k, w, images, labels, 6, smooth=True)
    frontal_du = af_Sobel.plotting_assd(dx, dy, mask, frontal_target_img, quiver=False, plot=False, display=False)

    sagittal_target_img = images[0:90, 0:512, sagittal_slice]
    sagittal_target_label = labels[..., 6][0:90, 0:512, sagittal_slice]
    dx, dy, mask, t, L = af_Sobel.assd_Sobel(sagittal_target_img, sagittal_target_label, voxelsize, a, SD, circles, seed, k, w, images, labels, 6, smooth=True)
    sagittal_du = af_Sobel.plotting_assd(dx, dy, mask, sagittal_target_img, quiver=False, plot=False, display=False)
    sagittal_du = np.where(sagittal_du < 0.5, 0, 4)
    
    blue_patch = mpatches.Patch(color='blue', label="Clinician's contour")
    red_patch = mpatches.Patch(color='red', label='DU contour')


    fig, ax = plt.subplots(2, 3, figsize=[12, 12])

    ax[0, 0].set_title("Tranverse (Horizonatal) slice")
    ax[0, 0].imshow(target_img5, cmap='gray', vmin=-250, vmax=250)
    ax[0, 0].contour(target_label5, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[0, 0].set_xlabel("Orginal contour")
    ax[0, 0].axis('off')

    ax[0, 1].set_title("Frontal (Coronal) slice")
    ax[0, 1].imshow(frontal_target_img.transpose(), cmap='gray', vmin=-250, vmax=250)
    ax[0, 1].contour(frontal_target_label.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[0, 1].set_xlabel("Orginal contour")
    ax[0, 1].axis('off')

    ax[0, 2].set_title("Sagittal (Longitudinal) slice")
    ax[0, 2].imshow(sagittal_target_img.transpose(), cmap='gray', vmin=-250, vmax=250)
    ax[0, 2].contour(sagittal_target_label.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[0, 2].set_xlabel("Orginal contour")
    ax[0, 2].axis('off')

    ax[1, 0].set_title("Tranverse (Horizonatal) slice")
    ax[1, 0].imshow(target_img5, cmap='gray', vmin=-250, vmax=250)
    ax[1, 0].contour(target_label5, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[1, 0].contour(du_5, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='red')
    ax[1, 0].set_xlabel("DU contour")
    ax[1, 0].set_xlim(200, 300)
    ax[1, 0].set_ylim(300, 200)
    ax[1, 0].legend(handles=[ blue_patch, red_patch])
    ax[1, 0].axis('off')

    ax[1, 1].set_title("Frontal (Coronal) slice")
    ax[1, 1].imshow(frontal_target_img.transpose(), cmap='gray', vmin=-250, vmax=250)
    ax[1, 1].contour(frontal_target_label.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[1, 1].contour(frontal_du.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='red')
    ax[1, 1].set_xlabel("DU contour")
    ax[1, 1].set_xlim(50, 100)
    ax[1, 1].set_ylim(300, 200)
    ax[1, 1].legend(handles=[ blue_patch, red_patch])
    ax[1, 1].axis('off')

    ax[1, 2].set_title("Sagittal (Longitudinal) slice")
    ax[1, 2].imshow(sagittal_target_img.transpose(), cmap='gray', vmin=-250, vmax=250)
    ax[1, 2].contour(sagittal_target_label.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='blue')
    ax[1, 2].contour(sagittal_du.transpose(), levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='red')
    ax[1, 2].set_xlabel("DU contour")
    ax[1, 2].set_xlim(50, 100)
    ax[1, 2].set_ylim(300, 200)
    ax[1, 2].legend(handles=[ blue_patch, red_patch])
    ax[1, 2].axis('off')

    plt.show()
    

