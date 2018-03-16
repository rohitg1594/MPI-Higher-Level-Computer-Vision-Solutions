import matplotlib.pyplot as plt
from vocab import create_vocab_tree
import cv2
import numpy as np
from utils import count_leaves

def Show_visual_word(path, folder):
    vis_to_desc, imgs, vocab_tree, descs, desc_arr, desc_to_kp, kps, _ = create_vocab_tree(folder)

    descs = vis_to_desc[tuple(path)]
    print(descs)

    counter = 1
    for desc in descs:
        img_i, kp_i = desc_to_kp[desc]
        print(img_i)
        plt.subplot(len(descs)//2 + 1,2,counter)
        counter += 1
        x , y = kps[img_i][kp_i].pt
        scale = 6*round(kps[img_i][kp_i].size)
        x, y = round(x), round(y)
        plt.imshow(imgs[img_i][x-scale:x+scale,y-scale:y+scale], cmap='gray')
    plt.show()


