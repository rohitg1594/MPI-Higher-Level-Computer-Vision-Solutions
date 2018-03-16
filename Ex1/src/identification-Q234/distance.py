'''Computes l2, intersection and chi-squared distances between two arrays'''
import numpy as np


def dist_l2(A, B):
    '''Computes the L2(euclidean) distance between two arrays'''

    return np.sum(np.square(A-B))

def dist_intersect(A, B):
    '''Computes the intersection of two histograms.
       intersect(A, B) = sum(min(A,B))'''

    return np.sum(np.minimum(A,B))

def dist_chi2(A, B, eps=10**-10):
    '''Computes the chi-squared distance between two arrays'''

    return np.sum(np.square(A-B)/(A+B+eps))
    
