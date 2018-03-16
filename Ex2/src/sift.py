import cv2

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
