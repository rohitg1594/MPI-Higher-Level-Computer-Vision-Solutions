from gauss import gauss_1d
from gaussdx import gaussdx
import part_c

import math
import numpy as np

import cv2
import matplotlib.pyplot as plt


def gaussianfilter(img, sigma):
    '''Outputs the result of applying 2D-gaussian filter of of sd sigma to the image.'''
    m, n = img.shape
    kernel, _ = gauss_1d(sigma)
    L = len(kernel)
    pad = L//2
    #pad along x-axis
    img = np.pad(img, ((0,0), (pad,pad)),'constant')
    kernel_x = np.tile(kernel,(m,1))
    plt.imshow(kernel_x)
    temp = np.zeros((m,n))
    #filter along x-axis
    for j in range(n):
        temp[:,j] = np.sum(img[:,j:j+L]*kernel_x)
    #pad along y-axis       
    temp = np.pad(temp, ((pad,pad), (0,0)),'constant')
    #plt.imshow(temp)
    #plt.show()
    kernel_y = np.tile(kernel.reshape(L,1), (1,n))
    print(kernel_y.shape)
    out = np.zeros((m,n))
    #filter along y-axis
    for i in range(m):
        out[i,:] = np.sum(temp[i:i+L,:]*kernel_y)
    return out


if __name__ == "__main__":
    #parts a and b
    img = cv2.imread('graf.png',0)
    filtered_image = gaussianfilter(img,4)
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(filtered_image, cmap='gray')
    plt.show()
    #part_c.part_c()
    

    