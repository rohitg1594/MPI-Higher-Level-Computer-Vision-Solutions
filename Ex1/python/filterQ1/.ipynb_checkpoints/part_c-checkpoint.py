from gauss import gauss_1d
from gaussdx import gaussdx

import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_filter(img, kernel, axis=0):
    m, n = img.shape
    L = len(kernel)
    pad = L
    out = np.zeros((m,n))
    
    if axis==0:
        img = np.pad(img, ((0,0), (pad,pad)),'constant')
        for i in range(m):
            for j in range(n):
                out[i,j] = np.sum(img[i,j:j+L]*kernel)
    else:
        img = np.pad(img, ((pad, pad), (0,0)),'constant')
        for i in range(m):
            for j in range(n):
                out[i,j] = np.sum(img[i:i+L,j]*kernel)
    return out

def part_c():
    sigma = 6
    imgImp = np.zeros((25, 25))
    imgImp[13, 13] = 1
    G, _ = gauss_1d(sigma)
    D, _ = gaussdx(sigma)
    
    quest_list = [((G,0),(G,1)),((G,0),(D,1)),((D,0),(G,1)),((G,1),(G,0)),((D,1),(G,0))]
    counter = 1
    for tups in quest_list:
        tup1, tup2 = tups
        kernel1, axis1 = tup1
        kernel2, axis2 = tup2
        
        img1 = apply_filter(imgImp, kernel1, axis=axis1)
        img2 = apply_filter(img1, kernel2, axis=axis2)
    
        plt.subplot(5,3,counter)
        plt.title('Original Image')
        plt.imshow(imgImp)
        counter += 1
        
        plt.subplot(5,3,counter)
        if np.array_equal(kernel1,G):
            plt.title('G{}'.format(axis1))
        if np.array_equal(kernel1,D):
            plt.title('D{}'.format(axis2))
        plt.imshow(img1)
        counter += 1
        
        plt.subplot(5,3,counter)
        if np.array_equal(kernel2,G):
            plt.title('G{}'.format(axis1))
        if np.array_equal(kernel2,D):
            plt.title('D{}'.format(axis2))
        plt.imshow(img2)
        counter += 1

    
    plt.tight_layout()
    plt.show()