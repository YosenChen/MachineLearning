#!/usr/bin/env python

'''
SVM and KNearest Chinese calligraphy style recognition.

First it loads a classified dataset of Chinese calligraphy images
Then it trains a SVM and KNearest classifiers on it and evaluates their accuracy

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Chinese character images are split into
   4 square cells (each of dimension loadImages.SZ/2) and 16-bin
   histogram of oriented gradients is computed for each cell
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
    if source==DataSource.csv:
        print('loading %s, label %s, ...' % (csv_path[style], style_dict[style][1]))
        imgList = loadData.loadCsv2Imgs(csv_path[style], rescale=False)
    elif source==DataSource.csv2:
        print('loading %s, label %s, ...' % (csv_path2[style], style_dict[style][1]))
        imgList = loadData.loadCsv2Imgs(csv_path2[style], rescale=False)
    else: raise Exception('unsupported source type')

    print('loaded %s images' % len(imgList))
    imgs = np.array(imgList)
    style_id = style_dict[style][1]
    labels = np.array([style_id for i in range(imgs.shape[0])])
    return imgs, labels

def load_all_styles(source):
    imgFeatLabList = []
    styles = ['kaishu', 'lishu', 'zhuanshu', 'caoshu', 'xingshu']
    for style in styles:
        imgs, labels = load_images_and_labels(style, source)
        imgFeatLabList.append((imgs, preprocess_hog(imgs), labels))
    return imgFeatLabList

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


def evaluate_model(model, imgs, samples, labels, prefix, vis_dim=0):
    if model is not None: resp = model.predict(samples)
    else: resp = samples  #FIXME: very bad practice, reuse samples as resp when model is None
    
    print('labels size: %s, resp size: %s'%(len(labels), len(resp)))

    err = (labels != resp).mean()
    print('%s error: %.2f %%' % (prefix, err*100))

    confusion = np.zeros((5, 5), np.float32)
    for i, j in zip(labels, resp):
        confusion[i-1, j-1] += 1.0
    print('%s confusion matrix:' % prefix)
    for i in range(confusion.shape[0]):
        confusion[i, :] = confusion[i, :]/sum(confusion[i, :])
    print(confusion)

    if vis_dim == 0: return None

    # bad practice, but no time
    kai_idx = np.array([0, 10, 20, 30, 40])
    li_idx  = np.array([0, 10, 20, 30, 40]) + 505
    zha_idx = np.array([7, 17, 22, 30, 40]) + 505 + 500
    cao_idx = np.array([8, 19, 20, 30, 40]) + 505 + 500 + 500
    xin_idx = np.array([0, 10, 20, 30, 40]) + 505 + 500 + 500 + 514
    
    vis_img = imgs[kai_idx]
    vis_img = np.concatenate((vis_img, imgs[li_idx]))
    vis_img = np.concatenate((vis_img, imgs[zha_idx]))
    vis_img = np.concatenate((vis_img, imgs[cao_idx]))
    vis_img = np.concatenate((vis_img, imgs[xin_idx]))
    
    resp = np.array(resp)
    vis_resp = resp[kai_idx]
    vis_resp = np.concatenate((vis_resp, resp[li_idx]))
    vis_resp = np.concatenate((vis_resp, resp[zha_idx]))
    vis_resp = np.concatenate((vis_resp, resp[cao_idx]))
    vis_resp = np.concatenate((vis_resp, resp[xin_idx]))
    
    lab = np.array(labels)
    vis_lab = lab[kai_idx]
    vis_lab = np.concatenate((vis_lab, lab[li_idx]))
    vis_lab = np.concatenate((vis_lab, lab[zha_idx]))
    vis_lab = np.concatenate((vis_lab, lab[cao_idx]))
    vis_lab = np.concatenate((vis_lab, lab[xin_idx]))
    
    vis = []
    for img, lab, res  in zip(vis_img, vis_lab, vis_resp):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img*(255.0/np.amax(img))
        img = img.astype(dtype=np.uint8)
        if lab!=res:
            if res==1: # R
                img[...,:2] = 0  # B& G off
            elif res==2: # G
                img[..., 0] = 0  # B off
                img[..., 2] = 0  # R off
            elif res==3: # GR
                img[..., 0] = 0  # B off
            elif res==4: # BG
                img[..., 2] = 0  # R off
            else: # BR
                img[..., 1] = 0  # G off
                
        vis.append(img)
    return mosaic(5, vis)

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

def loadPredResTxt(filename):
    res = []
    with open(filename) as f:
        for x in f.readlines():
            x = x.strip()
            res.append(int(x))
        print("load %s results from %s"%(len(res), filename))
        return np.array(res)
    return None

if __name__ == '__main__':
    print(__doc__)

    if len(sys.argv) != 2:
        print("Usage: python OpenCV_SVM_kNN_HOG.py <data source type 2(csv) or 3(csv2)>")
        exit()
    else:
        if int(sys.argv[1])==2:
            print('use DataSource.csv...')
            source = DataSource.csv
        elif int(sys.argv[1])==3:
            print('use DataSource.csv2...')
            source = DataSource.csv2
        else:
            print('unsupported data source type')
            exit()

    imgFeatLabList = load_all_styles(source)
    train_n = 1500
    for imgs, feat, lab in imgFeatLabList:
        imgs_train, imgs_test = np.split(imgs, [train_n])
        feats_train, feats_test = np.split(feat, [train_n])
        labels_train, labels_test = np.split(lab, [train_n])
        print("label:%s, #train:%s, #test:%s"%(lab[0], len(imgs_train), len(feats_test)))
        if lab[0]==1:
            all_train_imgs = imgs_train
            all_train_feats = feats_train
            all_train_labels = labels_train
            all_test_imgs = imgs_test
            all_test_feats = feats_test
            all_test_labels = labels_test
        else:
            all_train_imgs = np.concatenate((all_train_imgs, imgs_train))
            all_train_feats = np.concatenate((all_train_feats, feats_train))
            all_train_labels = np.concatenate((all_train_labels, labels_train))
            all_test_imgs = np.concatenate((all_test_imgs, imgs_test))
            all_test_feats = np.concatenate((all_test_feats, feats_test))
            all_test_labels = np.concatenate((all_test_labels, labels_test))

    print('training KNearest...')
    model = KNearest(k=4)
    model.train(all_train_feats, all_train_labels)
    vis = evaluate_model(model, all_train_imgs, all_train_feats, all_train_labels, 'kNN train')
    # cv2.imshow('KNearest train', vis)
    vis = evaluate_model(model, all_test_imgs, all_test_feats, all_test_labels, 'kNN test', 5)
    cv2.imshow('KNearest test', vis)

    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    model.train(all_train_feats, all_train_labels)
    vis = evaluate_model(model, all_train_imgs, all_train_feats, all_train_labels, 'SVM train')
    # cv2.imshow('SVM train', vis)
    vis = evaluate_model(model, all_test_imgs, all_test_feats, all_test_labels, 'SVM test', 5)
    cv2.imshow('SVM test', vis)

    # NOTE that txt result are all 0-based
    vis = evaluate_model(None, all_test_imgs, loadPredResTxt('test_result/random_forest_test.txt')+1, \
            all_test_labels, 'RF test', 5)
    cv2.imshow('RF test', vis)
    vis = evaluate_model(None, all_test_imgs, loadPredResTxt('test_result/softmax_HOG_train_0.968_test_0.955538.txt')+1, \
            all_test_labels, 'softmax HOG test', 5)
    cv2.imshow('softmax HOG test', vis)
    vis = evaluate_model(None, all_test_imgs, loadPredResTxt('test_result/softmax_train_0.853067_test_0.718936.txt')+1, \
            all_test_labels, 'softmax test', 5)
    cv2.imshow('softmax test', vis)

    
    cv2.waitKey(0)
