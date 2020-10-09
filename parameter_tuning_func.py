# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:45:35 2020

@author: lykha
"""
import numpy as np 
import pandas as pd 
import assd_func_Sobel_2 as af_Sobel
import matplotlib.pyplot as plt
#import assd_func as af

def create_log(df, c, k, w, ismax, dice):
    if ismax:
        ismax = "max"
    else: 
        ismax = "min"
    df = df.append({'c' : c, 'k' : k, 'w': w, 'min/max' : ismax, 'dice': dice},  
                ignore_index = True) 
    return df

def dice_coef(seg, gt, k=1):
    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def find_c(target_img1, target_label1, images, labels, c_list, xlim=[300, 190], ylim=[300, 190], plot_dice=True):
    dice_list = []
    a=50
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    circles = 3
    seed=123
    w = 2
    c_list = c_list
    k = 1
    df = pd.DataFrame(columns = ['k', 'c', 'w', 'min/max', "dice"]) 
    ismax=False
    plt.figure(figsize=(15, 15))
    for i in range(len(c_list)):
        c = c_list[i]
        SD=[c*1.7, c*2, c*2.5]
        dx, dy, mask, t1, L1, roi_z = af_Sobel.assd_Sobel(target_img1, target_label1, voxelsize, a, SD, circles, seed, k, w, images, labels, 1, ismax=ismax, smooth=True)
        du1 = af_Sobel.plotting_assd(dx, dy, mask, target_img1, quiver=False, plot=False)
        dice = dice_coef(du1, target_label1, 1)
        dice_list.append(dice)
        df = create_log(df, c, k, w, ismax, dice)
        plt.subplot(4, 5, i + 1)
        plt.imshow(target_img1, cmap="gray")
        plt.contour(mask, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="green")
        plt.contour(du1, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="red")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title("c = " + str(c_list[i]))
        plt.axis('off')
    # plot dice  
    if plot_dice:
        plt.figure(figsize=(10, 10))
        plt.plot(c_list, dice_list, label="Dice coef")
        plt.ylabel("Dice")
        plt.xlabel("c*[1.7, 2, 2.5]")
        plt.legend()
    return df

#Input is a list of paremeters for k and selected c
def find_k(target_img1, target_label1, images, labels, k_list, c, df, xlim=[300, 190], ylim=[300, 190], plot_dice=True):
    print(str(c) + "*[1.7, 2, 2.5]")
    dice_list = []
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    a=50
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    circles = 3
    seed=123
    w = 2
    c = c
    k_list = k_list
    #df = pd.DataFrame(columns = ['k', 'c', 'min/max', 'w', "dice"]) 
    ismax=False
    plt.figure(figsize=(15, 15))

    for i in range(len(k_list)):
        k = k_list[i]
        SD=[c*1.7, c*2, c*2.5]
        dx, dy, mask, t1, L1, roi_z = af_Sobel.assd_Sobel(target_img1, target_label1, voxelsize, a, SD, circles, seed, k, w, images, labels, 1, ismax=ismax, smooth=True)
        du1 = af_Sobel.plotting_assd(dx, dy, mask, target_img1, quiver=False, plot=False)
        dice = dice_coef(du1, target_label1, 1)
        dice_list.append(dice)
        df = create_log(df, c, k, w, ismax, dice)
        plt.subplot(4, 5, i + 1)
        plt.imshow(target_img1, cmap="gray")
        plt.contour(mask, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="green")
        plt.contour(du1, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="red")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title("k = " + str(k_list[i]))
        plt.axis('off')
    # plot dice  
    if plot_dice:
        plt.figure(figsize=(10, 10))
        plt.plot(k_list, dice_list, label="Dice coef")
        plt.ylabel("Dice")
        plt.xlabel("k")
        plt.legend()
    return df

#Input is a list of paremeters for w and selected k & c
def find_w(target_img1, target_label1, images, labels, w_list, k, c, df, xlim=[300, 190], ylim=[300, 190], plot_dice=True):
    print("SD: " + str(c) + "*[1.7, 2, 2.5]")
    print("k: " + str(k))
    dice_list = []
    a=50
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    circles = 3
    seed=123
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    w_list = w_list
    c = c
    k = k
    plt.figure(figsize=(15, 15))

    #df = pd.DataFrame(columns = ['k', 'c', 'w', 'min/max', "dice"]) 
    ismax=False
    for i in range(len(w_list)):
        w = w_list[i]
        SD=[c*1.7, c*2, c*2.5]
        dx, dy, mask, t1, L1, roi_z = af_Sobel.assd_Sobel(target_img1, target_label1, voxelsize, a, SD, circles, seed, k, w, images, labels, 1, ismax=ismax, smooth=True)
        du1 = af_Sobel.plotting_assd(dx, dy, mask, target_img1, quiver=False, plot=False)
        dice = dice_coef(du1, target_label1, 1)
        dice_list.append(dice)
        df = create_log(df, c, k, w, ismax, dice)
        plt.subplot(4, 5, i + 1)
        plt.imshow(target_img1, cmap="gray")
        plt.contour(mask, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="green")
        plt.contour(du1, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="red")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title("w = " + str(w_list[i]))
        plt.axis('off')
    # plot dice  
    if plot_dice:
        plt.figure(figsize=(10, 10))
        plt.plot(w_list, dice_list, label="Dice coef")
        plt.ylabel("Dice")
        plt.xlabel("w")
        plt.legend()
    return df

def find_ismax(target_img1, target_label1, images, labels, w, k, c, df, xlim=[300, 190], ylim=[300, 190], ismax=False, plot_dice=True):
    print("SD: " + str(c) + "*[1.7, 2, 2.5]")
    print("k: " + str(k))
    dice_list = []
    a=50
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    circles = 3
    seed=123
    voxelsize = np.array([0.976562, 0.976562, 2.5])
    w = w
    c = c
    k = k
    plt.figure(figsize=(15, 15))

    #df = pd.DataFrame(columns = ['k', 'c', 'w', 'min/max', "dice"]) 
    ismax=False
    for i in range(len(w_list)):
        SD=[c*1.7, c*2, c*2.5]
        dx, dy, mask, t1, L1, roi_z = af_Sobel.assd_Sobel(target_img1, target_label1, voxelsize, a, SD, circles, seed, k, w, images, labels, 1, ismax=ismax, smooth=True)
        du1 = af_Sobel.plotting_assd(dx, dy, mask, target_img1, quiver=False, plot=False)
        dice = dice_coef(du1, target_label1, 1)
        dice_list.append(dice)
        df = create_log(df, c, k, w, ismax, dice)
        plt.subplot(4, 5, i + 1)
        plt.imshow(target_img1, cmap="gray")
        plt.contour(mask, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="green")
        plt.contour(du1, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="red")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        if ismax:
            plt.title("ismax = True")
        else:
            plt.title("ismax = False")
        plt.axis('off')
    return df

