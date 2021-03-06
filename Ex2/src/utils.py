import glob
import numpy as np
import cv2
from collections import defaultdict

def load_images(folder):
    names = glob.glob('../data/' + folder + '/*.jpg')
    names = sorted(names)
    imgs_c = [cv2.imread(name) for name in names]
    imgs_g = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_c]

    return imgs_g, imgs_c


def concatenate_descriptors(descs):
    desc_arr = np.zeros((0,128))
    desc_to_kp = defaultdict(list)
    len_comp = 0
    for i, desc in enumerate(descs):
        for j in range(desc.shape[0]):
                desc_to_kp[j + len_comp] = [i, j]
        len_comp += len(desc)
        desc_arr = np.concatenate((desc_arr, desc), axis=0)

    return desc_arr, desc_to_kp


def count_leaves(vocab_tree):
    if len(vocab_tree.children) < 1:
        return len(vocab_tree.centers)
    else:
        count = 0
        for child in vocab_tree.children:
            count += count_leaves(child)
        return count


def draw_key_points(imgs, kps):    
    imgs_kp = [cv2.drawKeypoints(imgs_g[i], kps[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(kps))]
    counter = 1
    n = len(kps)
    for i in range(n):
        plt.subplot(n/2, 2, counter)
        counter += 1
        plt.imshow(imgs_kp[i])        
    plt.show()


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)
