from utils import load_images, concatenate_descriptors, closest_node
from sift import sift_detect_and_compute
from cyvlfeat.kmeans.kmeans import kmeans
import numpy as np
import cv2
import glob
import re


def create_codebook(folder, vocab_size=4096):
    imgs, _ = load_images(folder)
    imgs = imgs
    kps, descs, num_kps = sift_detect_and_compute(imgs)
    descs, desc_to_kp = concatenate_descriptors(descs)
    codebook = kmeans(descs, vocab_size)

    return codebook, desc_to_kp


def extract_bow_features(folder, codebook):
    names = glob.glob('../data/' + folder + '/*.jpg')
    names = sorted(names)
    
    pattern = r"([a-z]*)"
    labels = [re.match(pattern, name.split('/')[3]).group(0) for name in names]

    imgs_c = [cv2.imread(name) for name in names]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_c]

    sift = cv2.xfeatures2d.SIFT_create()    
    hist_arr = np.zeros((len(imgs), len(codebook)))
    for i, img in enumerate(imgs):
        kps, descs = sift.detectAndCompute(img, None)
        for desc in descs:
            code = closest_node(desc, codebook)
            hist_arr[i, code] += 1


    return hist_arr, labels
    
    
