import math
import numpy as np

def gauss_1d(sigma):
        '''Computes the 1-d gaussian for given standard-deviation
           outputs the values over vector [-3*sigma, 3*sigma ] and also
           return x vector.'''
        PI = np.pi
        x = np.arange(-3*sigma,3*sigma + 1)
        num = np.exp(-x*x/(2*sigma*sigma))
        den = np.sqrt(2*PI)*sigma
        return num/den, x
