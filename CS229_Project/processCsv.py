import pandas as pd
import numpy as np
import loadImages
import loadData
from OpenCV_SVM_kNN_HOG import preprocess_hog
import sys


def processHogOnCsv(srcName, dstName):
    print "process: %s..." % srcName
    imgs = loadData.loadCsv2Imgs(srcName, False)
    imgs = preprocess_hog(imgs)
    pd.DataFrame(imgs).to_csv(dstName, header=False, index=False)
    print "saved as %s." % dstName
    
if __name__ == "__main__":

    if len(sys.argv) == 4:
        loadImages.SZ = int(sys.argv[3])
        processHogOnCsv(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 3:
        loadImages.SZ = 96
        processHogOnCsv(sys.argv[1], sys.argv[2])
    else:
        print "Usage: python processCsv.py <srcCsvName> <dstCsvName> <optional: img size, default 96>"

