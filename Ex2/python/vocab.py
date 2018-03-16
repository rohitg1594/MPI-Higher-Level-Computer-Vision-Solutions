import numpy as np
from collections import defaultdict
import cyvlfeat.kmeans as vf

from utils import *
from sift import *


def create_vocab_tree(folder):
    # read images
    imgs_g, _ = load_images(folder)

    # detect keypoints and compute descriptors
    kps, descs, num_kps = sift_detect_and_compute(imgs_g)

    # concatenate all descriptors and create dictionary to map
    # from descriptor number to the image index and keypoint index
    desc_arr, desc_to_kp = concatenate_descriptors(descs)

    # fit the tree
    vocab_tree, assignments = vf.hikmeans(desc_arr.astype(np.uint8), 8, 4096)

    # make index from leaf to descriptors
    vis_to_desc = defaultdict(list)
    for i, assignment in enumerate(assignments):
        vis_to_desc[tuple(assignment)].append(i)


    return vis_to_desc, imgs_g, vocab_tree, descs, desc_arr, desc_to_kp, kps, num_kps

    

            





    
    
