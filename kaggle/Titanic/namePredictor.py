# please refer to Oct 5th lecture 
# and slides: learning3.pdf, page 45, Algorithm: recipe for success

import submission_hw2, util
from collections import defaultdict

# Read in examples
trainExamples = util.readExamples('names.train')
devExamples = util.readExamples('names.dev')

# SGD: train err = 0.001276, dev err = 0.19609, 3935 weights
# Comment: overfitting, so test error is very bad
# weights: (format of each row: <feature name> <feature value>) 
#   yes, all entities (w/ positive weights) are people's name
#   and when we look at the bottom (negative weights), those are not people's names
# error-analysis: score = dotProduct(weights, phi)
#   1st wrong result: 'Eduardo Romero' (truth = 1, our prediction = -1)
#       score = 0, because this name (Eduardo Romero) is not in the training set
#       clearly this is going to overfitting a lot, since we might have seen 
#       'Eduardo' and 'Romero' at different locations, 
#       but this extractor takes the whole name as a feature -> overfitting, need to break names down
def featureExtractor1(x):
    # x  = "took Mauritius into"
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + str(entity)] = 1
    return phi



# SGD: train err = 0.001787, dev err = 0.116558, 8543 weights 
# comment: test error has improved, almost a half
# error analysis: 
#   now Eduardo Romero has positive score (correct result)
#       'entity contains Eduardo'           has weight 0.27
#       'entity is ['Eduardo', 'Romero']'   has weight 0
#       'entity contains Romero'            has weight 0
#       since Eduardo did show up in training set, so overall we got positive score
#   1st wrong result: 'character Inspector Maigret ,' (truth = 1, our prediction = -1)
#       we didn't see 'Maigret' and 'Inspector' before, but we do have the context 'character'
#       maybe in the training set, someone uses 'character'
#       so we can add 2 feature templates left and right, then see how it goes
def featureExtractor2(x):
    # x  = "took Mauritius into"
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + str(entity)] = 1
    for word in entity:
        phi['entity contains ' + word] = 1
    return phi



# SGD: train err = 0.0100855, dev err = 0.063577, 10206 weights 
# comment: now the test error becomes lower!
# error analysis:
#   1st wrong result: '" Kurdistan Workers Party (' (truth = -1, our prediction = 1)
#       at this point, there's no kind of right answer or wrong answer, it's just soem intuition
#       it's not a bug in the code...
#       it's still probably overfitting, so we have to expand the entity further
#       break entity down into smaller pieces
def featureExtractor3(x):
    # x  = "took Mauritius into"
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + str(entity)] = 1
    phi['left is ' + left] = 1
    phi['right is ' + right] = 1
    for word in entity:
        phi['entity contains ' + word] = 1
    return phi



# SGD: train err = 0.0020426, dev err = 0.042640, 16651 weights
# comment: now the test err is even better!
def featureExtractor4(x):
    # x  = "took Mauritius into"
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + str(entity)] = 1
    phi['left is ' + left] = 1
    phi['right is ' + right] = 1
    for word in entity:
        phi['entity contains ' + word] = 1
        phi['entity prefix is ' + word[:4]] = 1 # take first 4 chars
        phi['entity suffix is ' + word[-4:]] = 1 # take last 4 chars
    return phi

# summary: 
#   we expand the entity (words, prefix, suffix), right, left to reduce overfitting
#   so we have more and more weights after training
#   if we don't break down x, just store the entity as a whole, 
#   we have less weights after training, 
#   but we overfit the training set, and have less generalization

# Learn a predictor
weights = submission_hw2.learnPredictor(trainExamples, devExamples, featureExtractor4, numIters=20, eta=0.01)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor4, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x: 1 if util.dotProduct(featureExtractor4(x), weights) > 0 else -1
print 'test error = ', util.evaluatePredictor(testExamples, predictor)

# comment: usually test err is probably a little higher than dev error

####################################################################################
"""
>>> a = 'hwo are you haha I dont care'
>>> a = a.split()
>>> a[-1]
'care'
>>> a[1:-1]
['are', 'you', 'haha', 'I', 'dont']
>>> a[0]
'hwo'
>>> str(a[1:-1])
"['are', 'you', 'haha', 'I', 'dont']"

"""
