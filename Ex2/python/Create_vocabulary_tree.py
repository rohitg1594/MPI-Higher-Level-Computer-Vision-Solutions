import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import numpy as np
from collections import defaultdict
import cyvlfeat.kmeans as vf

def sift_detect_and_compute(imgs):
    if not isinstance(imgs, list):
        imgs = []
        
    sift = cv2.xfeatures2d.SIFT_create()
    kps = []
    descs = []
    num_kps = []
    
    for img in imgs:
        kp, desc = sift.detectAndCompute(img, None)
        kps.append(kp)
        descs.append(desc)
        num_kps.append(len(kp))

    return kps, descs, num_kps


def create_vocabulary_tree(folder):
    # read images
    names = glob.glob('../data/' + folder + '/*.jpg')
    imgs_c = [cv2.imread(name) for name in names[0:100]]
    imgs_g = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_c]

    # detect keypoints and compute descriptors
    kps, descs, num_kps = sift_detect_and_compute(imgs_g)
        
    # concatenate all descriptors and create dictionary to map
    # from descriptor number to the image index and keypoint index
    desc_arr = np.zeros((0,128))
    desc_to_kp = defaultdict(list)
    len_comp = 0
    for i, desc in enumerate(descs):
        for j in range(desc.shape[0]):
                desc_to_kp[j + len_comp] = [i, j]
        len_comp += len(desc)
        desc_arr = np.concatenate((desc_arr, desc), axis=0)
    desc_arr = desc_arr[1:,:].astype(np.uint8)

    # fit the tree
    vocab_tree, assignments = vf.hikmeans(desc_arr, 8, 4096)

    # query the tree
    i = np.random.randint(0, len(desc_arr))
    path_to_leaf = np.asarray(vf.hikmeans_push(desc_arr, vocab_tree))[i]
    leaf = np.asarray(vocab_tree.children[path_to_leaf[0]].children[path_to_leaf[1]].children[path_to_leaf[2]].centers[path_to_leaf[3]])
    print(count_visual_words(vocab_tree))
    # create index of clusters to descriptors
    index_desc = defaultdict(list)
    for i, label in enumerate(vocab_tree.labels_):
        index_desc[label].append(i)

    # create index of clusters to images and scores
    index_images = create_img_index(index_desc)
    index_images = normalize(index_images, len(names), num_kps)
        
    return kps, vocab_tree, index_desc, index_images, desc_to_kp


def count_visual_words(vocab_tree):
    if len(vocab_tree.children) < 1:
        return len(vocab_tree.centers)
    else:
        count = 0
        for child in vocab_tree.children:
            count += count_visual_words(child)
        return count

    
def retrieval(img):
    _, vocab_tree, index_desc, index_images, _ = create_vocabulary_tree("Covers_train")

    kps, descs, num_kps = sift_detect_and_compute(img)

    
def create_img_index(index_desc):
    index_images = defaultdict(list)
    for clus, descs in index_desc.items():
        inside_dict = defaultdict(int)        
        for desc in descs:
            img_i = desc_to_kp[0]
            inside_dict[img_i] += 1
        index_images[clus] = inside_dict

    return index_images


def normalize(index_images, num_images, num_kps):
    N = num_images
    for clus, inside_dict in index_images.items():
        Ni = len(inside_dict)
        for img, score in inside_dict.items():
            score = (score/num_kps[img])*np.log(num_images/Ni)

    return index_images
            


def draw_key_points(imgs, kps):    
    imgs_kp = [cv2.drawKeypoints(imgs_g[i], kps[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(kps))]
    counter = 1
    n = len(kps)
    for i in range(n):
        plt.subplot(n/2, 2, counter)
        counter += 1
        plt.imshow(imgs_kp[i])        
    plt.show()


def Show_visual_word():
    kps, imgs, tree, index_desc, index_images, desc_to_kp = create_vocabulary_tree("Covers_test")
    descss = []
    for i in range(10):
        cno = np.random.randint(0, 4096)
        descs = index_desc[cno]
        descss.append(descs)
        print(descs)
    
    for descs in descss:
        counter = 1
        n = len(descs)
        plt.figure()
        for desc in descs:
            img_i, kp_i = desc_to_kp[desc]
            plt.subplot(n,1,counter)
            counter += 1
            x , y = kps[img_i][kp_i].pt
            scale = 6*round(kps[img_i][kp_i].size)
            x, y = round(x), round(y)
            plt.imshow(imgs[img_i][x-scale:x+scale,y-scale:y+scale], cmap='gray')
        plt.show()
    
    
if __name__ == "__main__":
    Show_visual_word()

