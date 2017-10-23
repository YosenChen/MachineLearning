import glob
import os
import cv2
import numpy as np

outSize = (96, 96)  # both dim must the same
SZ = outSize[0] # size of each digit is SZ x SZ

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    print img.shape
    return img

def loadImages(path, fileExt):
    if not os.path.isdir(path):
        raise RuntimeError('%s is not a valid folder' % path)
    print(os.path.join(path, fileExt))
    files = glob.glob(os.path.join(path, fileExt))
    imgList = []
    for file in files:
        # print('load image %s...'%file)
        img = cv2.imread(file, 0)
        img = cv2.resize(img,outSize, interpolation = cv2.INTER_LINEAR)
        imgList.append(img)
    return imgList

# cv doesn't support reading gif file
# a = loadImages('CC/giftype/1.kaishu', '*.gif')
# print a[1]

if __name__ == '__main__':
    print(__doc__)
    b = loadImages('CC/caoshu', '*.jpg')
    imgIdx = 0
    gx = cv2.Sobel(b[imgIdx], cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(b[imgIdx], cv2.CV_32F, 0, 1)
    print b[imgIdx]
    print b[imgIdx].shape  # height, width, channels = img.shape
    cv2.imshow('b[%s]'%imgIdx, b[imgIdx])
    cv2.imshow('b[%s]_deskew'%imgIdx, deskew(b[imgIdx]))
    cv2.imshow('gx', gx)
    cv2.imshow('gy', gy)
    
    cv2.waitKey(0)
