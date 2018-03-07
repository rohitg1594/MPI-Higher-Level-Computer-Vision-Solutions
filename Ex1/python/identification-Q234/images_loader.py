import glob
import sys
import cv2
import numpy as np

def sortKeyFunc(s):
    return int(s.split('.')[1].split('/')[2])

def load_images(folder,grayscale=False,num_query=10):

    if folder == 'model':
        names = sorted(glob.glob('./' + folder +'/*.png'), key=sortKeyFunc)
    elif folder == 'query':
        names = np.random.choice(glob.glob('./' + folder +'/*.png'),num_query)
    else:
        print('Wrong folder!')
        sys.exit(1)
        
    image_numbers = [int(name.split('.')[1].split('/')[2]) for name in names]

    if not grayscale:
        images = [cv2.imread(name) for name in names]
    else:
        images= [cv2.imread(name,0) for name in names]

    return image_numbers, images
        
