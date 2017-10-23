#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from '../data/digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   first enter cv virtual environment by: workon cv
   python OpenCV_SVM_kNN_HOG.py <optional: data source type>
'''


# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import numpy as np
from numpy.linalg import norm

import loadImages
import loadData

import sys

# local modules
from common import clock, mosaic

class DataSource(object):
    images = 1
    csv = 2
    csv2 = 3

style_dict = {  'kaishu':   ['CC/kaishu',   1], \
                'lishu':    ['CC/lishu',    2], \
                'zhuanshu': ['CC/zhuanshu', 3], \
                'caoshu':   ['CC/caoshu',   4], \
                'xingshu':  ['CC/xingshu',  5]}

csv_path = {    'kaishu':   'data_96_96/kaishu_96_96.csv', \
                'lishu':    'data_96_96/lishu_96_96.csv', \
                'zhuanshu': 'data_96_96/zhuanshu_96_96.csv', \
                'caoshu':   'data_96_96/caoshu_96_96.csv', \
                'xingshu':  'data_96_96/xingshu_96_96.csv'}

csv_path2 = {    'kaishu':   'data_96_bw/kaishu_96_bw.csv', \
                 'lishu':    'data_96_bw/lishu_96_bw.csv', \
                 'zhuanshu': 'data_96_bw/zhuanshu_96_bw.csv', \
                 'caoshu':   'data_96_bw/caoshu_96_bw.csv', \
                 'xingshu':  'data_96_bw/xingshu_96_bw.csv'}

def load_images_and_labels(style, source):
    if source==DataSource.images:
        print('loading %s, label %s, ...' % (style_dict[style][0], style_dict[style][1]))
        imgList = loadImages.loadImages(style_dict[style][0], '*.jpg')
    elif source==DataSource.csv:
        print('loading %s, label %s, ...' % (csv_path[style], style_dict[style][1]))
        imgList = loadData.loadCsv2Imgs(csv_path[style], rescale=False)
    elif source==DataSource.csv2:
        print('loading %s, label %s, ...' % (csv_path2[style], style_dict[style][1]))
        imgList = loadData.loadCsv2Imgs(csv_path2[style], rescale=False)
    else: raise Exception('unknown source type')
    print('loaded %s images' % len(imgList))
    imgs = np.array(imgList)
    style_id = style_dict[style][1]
    labels = np.array([style_id for i in range(imgs.shape[0])])
    return imgs, labels

def load_all_styles(source):
    all_imgs = None
    all_labels = None
    for i, style in enumerate(style_dict.keys()):
        imgs, labels = load_images_and_labels(style, source)
        if i==0:
            all_imgs = imgs
            all_labels = labels
        else:
            all_imgs = np.concatenate((all_imgs, imgs))
            all_labels = np.concatenate((all_labels, labels))
    return all_imgs, all_labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*loadImages.SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (loadImages.SZ, loadImages.SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/Itseez/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, imgs, samples, labels, prefix):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('%s error: %.2f %%' % (prefix, err*100))

    confusion = np.zeros((5, 5), np.float32)
    for i, j in zip(labels, resp):
        confusion[i-1, j-1] += 1.0
    print('%s confusion matrix:' % prefix)
    for i in range(confusion.shape[0]):
        confusion[i, :] = confusion[i, :]/sum(confusion[i, :])
    print(confusion)

    vis = []
    for img, flag in zip(imgs, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(15, vis)

def preprocess_simple(imgs):
    return np.float32(imgs).reshape(-1, loadImages.SZ*loadImages.SZ) / 255.0

def preprocess_hog(imgs):
    samples = []
    # print("img shape ", imgs[0].shape, loadImages.SZ)
    for img in imgs:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:int(loadImages.SZ/2),:int(loadImages.SZ/2)], \
                    bin[int(loadImages.SZ/2):,:int(loadImages.SZ/2)], \
                    bin[:int(loadImages.SZ/2),int(loadImages.SZ/2):], \
                    bin[int(loadImages.SZ/2):,int(loadImages.SZ/2):]
        mag_cells = mag[:int(loadImages.SZ/2),:int(loadImages.SZ/2)], \
                    mag[int(loadImages.SZ/2):,:int(loadImages.SZ/2)], \
                    mag[:int(loadImages.SZ/2),int(loadImages.SZ/2):], \
                    mag[int(loadImages.SZ/2):,int(loadImages.SZ/2):]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        # print("hist shape ", hist.shape)

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print(__doc__)
    print("Usage: python OpenCV_SVM_kNN_HOG.py <optional: data source type>")

    if len(sys.argv) != 2:
        print('use DataSource.images...')
        source = DataSource.images
    else:
        if int(sys.argv[1])==1:
            print('use DataSource.images...')
            source = DataSource.images
        elif int(sys.argv[1])==2:
            print('use DataSource.csv...')
            source = DataSource.csv
        else:
            print('use DataSource.csv2...')
            source = DataSource.csv2

    imgs, labels = load_all_styles(source)

    print('preprocessing...')
    # shuffle imgs
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(imgs))
    imgs, labels = imgs[shuffle], labels[shuffle]

    if source!=DataSource.images: print('skip deskew...')
    imgs2 = list(map(deskew, imgs)) if source==DataSource.images else imgs
    samples = preprocess_hog(imgs2)

    train_n = int(0.9*len(samples))
    cv2.imshow('test set', mosaic(15, imgs[train_n:]))
    imgs_train, imgs_test = np.split(imgs2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('training KNearest...')
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, imgs_train, samples_train, labels_train, 'kNN train')
    # cv2.imshow('KNearest train', vis)
    vis = evaluate_model(model, imgs_test, samples_test, labels_test, 'kNN test')
    cv2.imshow('KNearest test', vis)

    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, imgs_train, samples_train, labels_train, 'SVM train')
    # cv2.imshow('SVM train', vis)
    vis = evaluate_model(model, imgs_test, samples_test, labels_test, 'SVM test')
    cv2.imshow('SVM test', vis)
    print('saving SVM as "imgs_svm.dat"...')
    model.save('imgs_svm.dat')

    cv2.waitKey(0)
