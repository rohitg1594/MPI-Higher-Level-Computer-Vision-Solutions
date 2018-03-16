import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

import distance
import hist
from images_loader import load_images

def rpc_curve(num_query, num_bins, hist_func, dist_func, eps=10**-8):

    model_numbers, model_images = load_images('model', num_query=num_query)
    query_numbers, query_images = load_images('query', num_query=num_query)
    
    distances =  np.zeros((len(model_images), len(query_images)))
    for i, model_image in enumerate(model_images):
        for j, query_image in enumerate(query_images):
            model_hist, _ = hist_func(model_image, num_bins)
            query_hist, _ = hist_func(query_image, num_bins)
            distances[i, j] = distance.dist_chi2(model_hist, query_hist)

    taus = np.linspace(0.25, 1.5, 100)
    precision = np.zeros(len(taus))
    recall = np.zeros(len(taus))
    
    for i, tau in enumerate(taus):
        indices = np.where(distances<tau)
        predictions = defaultdict(list)
        for j in range(len(indices[1])):
            predictions[query_numbers[indices[1][j]]-1].append(indices[0][j])
            
        tp, fp, tn, fn= 0, 0, 0, 0
        for key, values in predictions.items():
            if key in values:
                tp += 1
                fp += len(values) - 1
                tn += len(model_images) - len(values)
            else:
                fp += len(values)
                tn += len(model_images) -len(values) - 1
                fn += 1

        # for when none of the predicted images are under tau
        if len(predictions.items()) != len(query_numbers):
            diff = len(query_numbers) - len(predictions.items())
            tn += (len(model_images) - 1)*diff
            fn += diff

        precision[i] = tp/(fp + tp + eps)
        recall[i] = tp/(tp + fn + eps)

    return precision, recall



if __name__ == '__main__':
    num_query = int(sys.argv[1])
    num_bins = int(sys.argv[2])

    chi2 = distance.dist_chi2
    intersect = distance.dist_intersect
    l2 = distance.dist_l2

    rgb = hist.rgb_hist
    rg = hist.rgb_hist
    
    pre_chi2_rgb, rec_chi2_rgb = rpc_curve(num_query, num_bins, rgb, chi2)
    pre_int_rgb, rec_int_rgb = rpc_curve(num_query, num_bins, rgb, intersect)
    pre_l2_rgb, rec_l2_rgb = rpc_curve(num_query, num_bins, rgb, l2)
 
    pre_chi2_rg, rec_chi2_rg = rpc_curve(num_query, num_bins, rg, chi2)
    pre_int_rg, rec_int_rg = rpc_curve(num_query, num_bins, rg, intersect)
    pre_l2_rg, rec_l2_rg = rpc_curve(num_query, num_bins, rg, l2)

    plt.subplot(121)
    plt.plot(1-pre_chi2_rgb, rec_chi2_rgb, label='chi2')
    plt.plot(1-pre_int_rgb, rec_int_rgb, label='int')
    plt.plot(1-pre_l2_rgb, rec_l2_rgb, label='l2')
    plt.legend()
    plt.title('RGB')

    plt.subplot(122)
    plt.plot(1-pre_chi2_rg, rec_chi2_rgb, label='chi2')
    plt.plot(1-pre_int_rg, rec_int_rgb, label='int')
    plt.plot(1-pre_l2_rg, rec_l2_rgb, label='l2')
    plt.legend()
    plt.title('RG')

    plt.show()
