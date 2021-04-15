import numpy as np
import cv2
import time
from source.utils import load_original_data
from tqdm import tqdm

def get_RGB_mean(img: np.array):
    assert img.shape[-1] == 3
    return img.mean(axis=(0,1))

def count_pixels_range_(img:np.array, range_: tuple, compute_mean=True):
    '''
    Given the array of an image with RGB channels and the range,
    return the count (or mean) of the designated range for 3 channels
    '''
    assert img.shape[-1] == 3
    condi1 = img >= range_[0]
    condi2 = img <  range_[1]
    out = np.mean(condi1 & condi2, axis=(0,1)) if compute_mean else np.sum(condi1 & condi2, axis=(0,1))
    return out

def count_pixels_Ranges_(img:np.array, n_Ranges:int, compute_mean=True):
    '''
    Given the array of an image with RGB channels and the no. of range (bin cut),
    return the count (or mean) of the designated ranges for 3 channels
    '''
    Ranges_=list(range(0, 255 + 255//n_Ranges, 255//n_Ranges))
    assert img.shape[-1] == 3
    assert len(Ranges_) >= 3
    assert min(Ranges_) >= 0
    
    out = np.array([])
    for i in range(len(Ranges_)-1):
        a = count_pixels_range_(img, (Ranges_[i], Ranges_[i+1]))
        out = np.concatenate([out, a])
    return out

def get_SIFT(img: np.array, compute_mean=True):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if compute_mean:
        descriptors = descriptors.mean(axis=0)
    return descriptors

def get_SURF(img: np.array, compute_mean=True):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(img, None)
    if compute_mean:
        descriptors = descriptors.mean(axis=0)
    return descriptors

def get_features_for_images(img_list, feature_type, n_Ranges=5):
    out = []
    for img in tqdm(img_list):
        if feature_type == 'Histogram':
            out.append(count_pixels_Ranges_(img, n_Ranges=n_Ranges))
        elif feature_type == 'SIFT':
            try:
                out.append(get_SIFT(img))
            except:
                out.append(np.array([0]*128))
        elif feature_type == 'SURF':
            try:
                out.append(get_SURF(img))
            except:
                out.append(np.array([0]*64))
        X = np.array(out)
    return X
