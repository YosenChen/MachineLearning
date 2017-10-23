import glob
import os
import cv2
import pandas as pd
import numpy as np

def scaleVec(v, valMax = 255.0):
    origMax = 23 # max(v) # almost all are 23
    # print "origMax: ", origMax
    for i in range(len(v)):
        v[i] = int(float(v[i])*valMax/origMax)
    return v

def loadCsv2Imgs(fname, rescale):
    csv = pd.read_csv(fname, header=None)
    print 'csv.size=%s, rescale=%s' % (csv.size, rescale)
    imgList = []
    for i in range(len(csv)):
        # print('load image %s...'%i)
        vec = np.array(csv.iloc[i, :])
        if rescale: vec = scaleVec(vec)
        # print vec.shape
        img = np.reshape(vec, (96, 96))
        img = img.astype(dtype=np.uint8)
        img = np.transpose(img)
        imgList.append(img)
    return imgList

if __name__ == '__main__':
    imgIdx = 1
    #a = loadCsv2Imgs('data_96_96/caoshu_96_96.csv', rescale=True)
    a = loadCsv2Imgs('data_96_bw/caoshu_96_bw.csv', rescale=False)

    gx = cv2.Sobel(a[imgIdx], cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(a[imgIdx], cv2.CV_32F, 0, 1)
    print a[imgIdx]
    print a[imgIdx].shape  # height, width, channels = img.shape
    cv2.imshow('a[%s]'%imgIdx, a[imgIdx])
    cv2.imshow('gx', gx)
    cv2.imshow('gy', gy)

    cv2.waitKey(0)

