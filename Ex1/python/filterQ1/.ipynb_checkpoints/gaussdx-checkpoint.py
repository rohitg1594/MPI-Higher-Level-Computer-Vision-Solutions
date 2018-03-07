import math
import numpy as np

def gaussdx(sigma):
        '''Computes the 1-d gaussian derivative filter for given standard-deviation,
           outputs the values over vector [-3*sigma, 3*sigma ] and also
           returns x vector.'''
        PI = np.pi
        x = np.arange(-3*sigma,3*sigma + 1)
        num = -x*np.exp(-x*x/(2*sigma*sigma))
        den = np.sqrt(2*PI)*sigma**3
        return num/den, x
