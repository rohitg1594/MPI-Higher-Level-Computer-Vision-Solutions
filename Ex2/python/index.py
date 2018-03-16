from vocab import create_vocab_tree
from collections import defaultdict
import numpy as np
from utils import *
import cyvlfeat.kmeans as vf
from sift import *
import matplotlib.pyplot as plt
import operator

def Invert_file_index(folder):
    vis_to_desc, imgs, vocab_tree, descs, desc_arr, desc_to_kp, kps, num_kps = create_vocab_tree(folder)
    
    # create index of visual words to images and scores
    index_images = create_img_index(vis_to_desc, desc_to_kp)
    index_images = normalize(index_images, len(imgs), num_kps)
        
    return imgs, index_images, vocab_tree

    
def create_img_index(vis_to_desc, desc_to_kp):
    index_images = defaultdict(dict)
    for vis, descs in vis_to_desc.items():
        inside_dict = defaultdict(int)        
        for desc in descs:
            img_i = desc_to_kp[desc][0]
            inside_dict[img_i] += 1
        index_images[vis] = dict(inside_dict)

    return dict(index_images)


def normalize(index_images, num_images, num_kps):
    N = num_images
    index = index_images
    for vis, inside_dict in index_images.items():
        Ni = len(inside_dict)
        for img, score in inside_dict.items():
            score = (score/num_kps[img])*np.log(num_images/Ni)
            inside_dict[img] = score

    return index_images


def retrieval(train_folder, img_q):
    train_imgs, index, vocab_tree = Invert_file_index(train_folder)
    sift = cv2.xfeatures2d.SIFT_create()
    kps, descs = sift.detectAndCompute(img_q, None)
    descs = descs.astype(np.uint8)
    paths = vf.hikmeans_push(descs, vocab_tree)
    
    img_score = defaultdict(int)    
    for i, path in enumerate(paths):
        img_and_score_dict = index[tuple(path)]
        for can_img, score in img_and_score_dict.items():
            img_score[can_img] += score

    sorted_img_score = sorted(img_score.items(), key=operator.itemgetter(1), reverse=True)

    
    counter = 1
    rows = 5
    cols = 2
    plt.subplot(rows,cols,counter)
    plt.imshow(img_q, cmap='gray')
    plt.title('QUERY IMAGE')
    print('CANDIDATES GENERATED:')
    for img_i, score in sorted_img_score[:rows*cols-1]:
        
        print('Index : {}, Score : {}'.format(img_i, score))
        counter += 1
        plt.subplot(rows, cols, counter)
        plt.imshow(train_imgs[img_i], cmap='gray')
        plt.title(str(img_i))

    plt.show()
