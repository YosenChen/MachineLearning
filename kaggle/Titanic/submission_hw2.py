#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    words = x.split()
    hist = {}
    for w in words:
        cnt = hist.get(w)
        hist[w] = 1 if cnt is None else (cnt+1)
    # print hist
    return hist
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for i in range(numIters):
        for x,y in trainExamples:
            # w = w - eta*grad_loss
            phi = featureExtractor(x)
            if dotProduct(phi, weights)*y < 1:
                increment(weights, eta*y, phi)
        trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print('iter#%s: trainErr = %s, testErr = %s' % (i, trainError, testError))
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        keys = random.sample(weights.keys(), random.randint(1, len(weights)))
        for k in keys:
            phi[k] = random.randint(1, 20)
        dp = dotProduct(phi, weights)
        y = 1 if dp > 0 else -1 if dp < 0 else 0
        (phi, y) = (phi, y) if y != 0 else generateExample()
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.split()
        s = ''
        for w in x: s += w
        res = {}
        # last n-gram is s[len(s)-n:len(s)], so i_max = len(s)-n, range upper bound = len(s)-n+1
        ngrams = [s[i:i+n] for i in range(len(s)-n+1)]
        for w in ngrams:
            val = res.get(w)
            res[w] = 1 if val is None else (val+1)
        return res
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
     
    # END_YOUR_CODE
