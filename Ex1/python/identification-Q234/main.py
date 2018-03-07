import sys

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

from retrieval import find_best_match
from images_loader import load_images

def showHighestMatch(query_images, predicted_images, cols=4):
    m = len(query_images)
    counter = 1
    for i in range(m):
        #plot the query image
        plt.subplot(m,cols,counter)
        counter += 1
        plt.imshow(query_images[i])
        #plot the nearest image
        plt.subplot(m,cols,counter)
        counter += 1
        plt.imshow(predicted_images[i])    
    plt.show()

if __name__ == "__main__":
    num_bins = int(sys.argv[1])
    num_query = int(sys.argv[2])
        
    _, model_images_color, model_images_gray = load_images('model')
    query_numbers, query_images_color, query_images_gray = load_images('query',num_query)

    dist_types = ['l2',  'chi2']
    hist_types = ['rgb', 'rg']
    recognition_rate = {}

    for dist_type in dist_types:
        for hist_type in hist_types:
            if hist_type in ['grayvalue', 'dxdy']:
                model_images, query_images = model_images_gray, query_images_gray
            else:
                model_images, query_images = model_images_color, query_images_color
                
            indices, distance = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
            predicted_images = [model_images[i] for i in indices]
            indices += 1                
            recognition_rate[dist_type +',' + hist_type] = np.sum(np.array(query_numbers)==indices)/len(query_images)
            
            print('True files:')
            print(np.array(query_numbers))
            print('Predicted files:')
            print(indices)
            print('Dist_type:{}, Hist_type:{} - {}'.format(dist_type, hist_type, recognition_rate[dist_type + ',' + hist_type]))
            showHighestMatch(query_images, predicted_images)


