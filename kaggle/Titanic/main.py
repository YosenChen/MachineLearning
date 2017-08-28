from collections import defaultdict
import os
import sys

"""
Note:
1. deal with those partially missing features:
   it could be a good idea for using binary values
   since 0 could be a very "wrong" guess... for the hidden true value
   ex: true age values might have 2 groups
   [15, 30] and [45, 60] and this grouping is strongly correlated to Survived value
   then using Age=0 is a terrible default value for missing ages
   for this case, use (0 if age=='' else 1 if int(age) > 25 else 0)
   could be a better idea.
   Or, maybe making defualt value = average value can be a better idea
2. 
"""

## reuse this function
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

## reuse this function
def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def extract_feature_1(keys, values):
    y = 1 if values[1]=='1' else 0
    x = defaultdict(float)
    # PassengerId
    x[keys[0]] = 1 if int(values[0])/2 else 0
    # Pclass
    x[keys[2]] = 0 if values[2]=='' else int(values[2])
    # Name (can be well extended)
    x[keys[3]] = len(values[3])
    # Sex
    x[keys[4]] = 1 if values[4]=='male' else 0
    # Age
    x[keys[5]] = 0 if values[5]=='' else 1 if int(values[5])>25 else 0
    # SibSp
    x[keys[6]] = 0 if values[6]=='' else int(values[6])
    # Parch
    x[keys[7]] = 0 if values[7]=='' else int (values[7])
    # Ticket
    x[keys[8]] = 0
    # Fare
    x[keys[9]] = 0
    # Cabin
    x[keys[10]] = 0
    # Embarked
    x[keys[11]] = 0
    return x, y


def extract_train_data(file_path):
    data_pairs = []  # (x, y) pairs
    keys = None
    for line in open(file_path):
        split_text = line.strip().split(',')
        
        if keys is None:  # split_text are keys
            keys = split_text
            print "keys: %s" % keys
            continue

        values = split_text
        x, y = extract_feature_1(keys, values)
        data_pairs.append((x, y))
    
    print "len(data_pairs) = %d" % len(data_pairs)
    return data_pairs


def extract_test_data(file_path):
    pass


## reuse this function
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



data_pairs = extract_train_data("train.csv")



