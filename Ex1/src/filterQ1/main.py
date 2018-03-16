from gaussfilter import *

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys

if __name__ == "__main__":
    print(sys.argv)
    
    if sys.argv[1] == 'a':
        img = cv2.imread('graf.png',0)
        filtered_image = gaussianfilter(img,4)
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(filtered_image, cmap='gray')
        plt.show()    

    if sys.argv[1] == 'c':
        part_c()

    if sys.argv[1] == 'd':
        img1 = cv2.imread('graf.png', 0)
        img2 = cv2.imread('gantrycrane.png', 0)
        sigma = 4
        deriv_x1, deriv_y1 = gaussderiv(img1, sigma)
        deriv_x2, deriv_y2 = gaussderiv(img2, sigma)

        plt.subplot(2,3,1)
        plt.imshow(img1, cmap='gray')
        plt.subplot(2,3,2)
        plt.imshow(deriv_x1, cmap='gray')
        plt.subplot(2,3,3)
        plt.imshow(deriv_y1, cmap='gray')
        plt.subplot(2,3,4)
        plt.imshow(img2, cmap='gray')
        plt.subplot(2,3,5)
        plt.imshow(deriv_x2, cmap='gray')
        plt.subplot(2,3,6)
        plt.imshow(deriv_y2, cmap='gray')

        plt.show()
