'''Implements retrieval functions find_best_match and show_neighbours'''

import distance
import hist

import numpy as np
import matplotlib.pyplot as plt

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):
    '''Returns the best match between the query_images and model_images.
       Input:
       model_images - list of file names of model images
       query_images - list of file names of query images

       dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
       hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'

       output:
       indices: array as same size as query_images with corresponding best match
       distances : 2D array of size(model_images)*size(query_images)'''
    if dist_type == 'l2':
        dist_func = distance.dist_l2
    elif dist_type == 'intersect':
        dist_func = distance.dist_intersect
    elif dist_type == 'chi2':
        dist_func = distance.dist_chi2
    else:
        print('Invalid distance measure')
        return None

    if hist_type == 'grayvalue':
        hist_func = hist.normalized_hist
    elif hist_type == 'dxdy':
        hist_func = hist.dxdy_hist
    elif hist_type == 'rgb':
        hist_func = hist.rgb_hist
    elif hist_type == 'rg':
        hist_func = hist.rg_hist
    else:
        print('Invalid hist type')
        return None

    #compute the distances matrix
    distances =  np.zeros((len(model_images), len(query_images)))
    for i, model_image in enumerate(model_images):
        for j, query_image in enumerate(query_images):
            model_hist, _ = hist_func(model_image, num_bins)
            query_hist, _ = hist_func(query_image, num_bins)
            distances[i, j] = dist_func(model_hist, query_hist)
    indices = np.argmin(distances,axis=0)

    return indices, distances


def show_neighbours(model_images, query_images, dist_type, hist_type, num_bins, num_neighbours=5):
    '''Displays the closest neighbours for query_images in model_images'''
    _, distances = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    num_meighbours = int(num_neighbours)
    topk = np.argsort(distances, axis=0)[:num_neighbours,:]   
    m = len(query_images)
    counter = 1
    for i in range(m):
        #plot the query image
        plt.subplot(m,num_neighbours+1,counter)
        counter += 1
        plt.imshow(query_images[i])
        plt.title('Query Image')
        for j in range(num_neighbours):
            #plot the nearest image
            plt.subplot(m,num_neighbours+1,counter)
            counter += 1
            plt.imshow(model_images[topk[j,i]])

    plt.tight_layout()
    plt.show()
