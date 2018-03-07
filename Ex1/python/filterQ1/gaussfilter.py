import math
import numpy as np

import matplotlib.pyplot as plt
import cv2

def gaussianfilter(img, sigma):
    '''Outputs the result of applying 2D-gaussian filter of of sd sigma to the image.'''
    m, n = img.shape
    kernel, _ = gauss_1d(sigma)
    L = len(kernel)
    pad = L//2
    
    #pad along x-axis
    img = np.pad(img, ((0,0), (pad,pad)),'constant')
    temp = np.zeros((m,n))
    kernel_x = np.tile(kernel, (m,1))
    #filter along x-axis
    for j in range(n):
        temp[:,j] = np.sum(img[:,j:j+L]*kernel_x,axis=1)

    #pad along y-axis       
    temp = np.pad(temp, ((pad,pad), (0,0)),'constant')
    out = np.zeros((m,n))
    kernel_y = np.tile(kernel.reshape(L,1), (1,n))
    #filter along y-axis
    for i in range(m):
        out[i,:] = np.sum(temp[i:i+L,:]*kernel_y,axis=0)
    return out


def gauss_1d(sigma):
        '''Computes the 1-d gaussian for given standard-deviation
           outputs the values over vector [-3*sigma, 3*sigma ] and also
           return x vector.'''
        PI = np.pi
        x = np.arange(-3*sigma,3*sigma + 1)
        num = np.exp(-x*x/(2*sigma*sigma))
        den = np.sqrt(2*PI)*sigma
        return num/den, x


def gaussdx(sigma):
        '''Computes the 1-d gaussian derivative filter for given standard-deviation,
           outputs the values over vector [-3*sigma, 3*sigma ] and also
           returns x vector.'''
        PI = np.pi
        x = np.arange(-3*sigma,3*sigma + 1)
        num = -x*np.exp(-x*x/(2*sigma*sigma))
        den = np.sqrt(2*PI)*sigma**3
        return num/den, x


def gaussderiv(img, sigma):
    '''Returns the gaussian derivative filters in x and y direction.'''
    m, n = img.shape
    kernel, _ = gaussdx(sigma)
    L = len(kernel)
    pad = L//2
    deriv_x = np.zeros_like(img)
    deriv_y = np.zeros_like(img)
    imgx_pad = np.pad(img, ((0,0), (pad, pad)), 'constant')
    imgy_pad = np.pad(img, ((pad,pad), (0, 0)), 'constant')
    kernel_x = np.tile(kernel, (m,1))
    kernel_y = np.tile(kernel.reshape(L,1),(1,n))    
    for j in range(n):
        deriv_x[:,j] = np.sum(imgx_pad[:,j:j+L]*kernel_x,axis=1)

    for i in range(m):
        deriv_y[i,:] = np.sum(imgy_pad[i:i+L,:]*kernel_y,axis=0)

    return deriv_x, deriv_y


def apply_filter(img, kernel, axis=0):
    '''Applies 1D kernel on image in the given direction'''
    m, n = img.shape
    L = len(kernel)
    pad = L
    out = np.zeros((m,n))
    
    if axis==0:
        img = np.pad(img, ((0,0), (pad,pad)),'constant')
        kernel_x = np.tile(kernel, (m,1))
        for j in range(n):
            out[:,j] = np.sum(img[:,j:j+L]*kernel_x,axis=1)
    else:
        img = np.pad(img, ((pad, pad), (0,0)),'constant')
        kernel_y = np.tile(kernel.reshape(L,1),(1,n))
        for i in range(m):
            out[i,:] = np.sum(img[i:i+L,:]*kernel_y,axis=0)
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
