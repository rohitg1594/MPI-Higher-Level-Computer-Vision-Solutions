'''Implements functions normalized_hist, rgb_hist,rg_hist and dxdy_hist.
   Covers parts a and b'''

import cv2
import matplotlib.pyplot as plt

import numpy as np

import sys

sys.path.insert(0, '/home/rohit/Documents/Spring_2018/Higher_level_computer_vision/Exercises/Ex1/python/filterQ1')

import gaussfilter


def normalized_hist(img, num_bins):
    '''Returns normalized histogram of pixel intensities of a gray-scale image.'''
    img = img.ravel()
    bin_edges = np.linspace(np.min(img), np.max(img), num_bins)
    inds = np.digitize(img, bin_edges)
    hist = np.zeros(num_bins)
    
    for i in range(len(img)):
        hist[inds[i]-1] += 1
    hist /= np.sum(hist)
    return hist, bin_edges


def rgb_hist(img, num_bins):
    '''Returns a normalized 3D histogram for a color image'''
    hist = np.zeros((num_bins,num_bins,num_bins))
    m, n, k = img.shape

    img_r = img[:,:,0].ravel()
    img_g = img[:,:,1].ravel()
    img_b = img[:,:,2].ravel()

    bin_edges = np.linspace(np.min(img), np.max(img), num_bins)

    inds_r = np.digitize(img_r, bin_edges)
    inds_g = np.digitize(img_g, bin_edges)
    inds_b = np.digitize(img_b, bin_edges)

    for i in range(len(img_r)):
        hist[inds_r[i]-1,inds_g[i]-1,inds_b[i]-1] += 1
    hist /= np.sum(hist)
    return hist, bin_edges


def rg_hist(img, num_bins):
    '''Returns a normalized 2D r and g values histogram.
       Accepts color image.'''
    hist = np.zeros((num_bins, num_bins))
    m, n, k = img.shape

    intensity = img.sum(axis=2).ravel()
    img_r = img[:,:,0].ravel()/intensity
    img_g = img[:,:,1].ravel()/intensity

    bin_edges = np.linspace(np.min(img), np.max(img), num_bins)

    inds_r = np.digitize(img_r, bin_edges)
    inds_g = np.digitize(img_g, bin_edges)

    for i in range(len(img_r)):
        hist[inds_r[i]-1,inds_g[i]-1] += 1
    hist /= np.sum(hist)
    return hist, bin_edges


def dxdy_hist(img, num_bins):
    '''Returns a histogram of first partial derivatives in the x and y direction.
       Only accepts grayscale images.'''
    deriv_x, deriv_y = gaussfilter.gaussderiv(img, 6)
    bin_edges = np.linspace(-34, 34, num_bins)

    hist = np.zeros((num_bins, num_bins))
    
    inds_x = np.digitize(deriv_x, bin_edges)
    inds_y = np.digitize(deriv_y, bin_edges)

    for i in range(len(deriv_x)):
        hist[inds_x[i]-1,inds_y[i]-1] += 1
    hist /= np.sum(hist)

    return hist, bin_edges
