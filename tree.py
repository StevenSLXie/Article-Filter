'''
Created on Dec 9, 2012
 
@author: Steven SL Xie
'''
from numpy import *
from numpy import linalg as la
from math  import *
from random import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import csv



def loadDataSet(fileName,str):      #general function to parse tab -delimited floats
	dataMat = []                #assume last column is target value
	with open(fileName) as fr:
		next(fr)
		for line in fr:
			curLine = line.strip().split(str)
			curLine = map(float,curLine) #map all elements to float()
			dataMat.append(curLine)
	return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
#print dataSet
	return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
#	print S,bestS
    if (S - bestS) < tolS: 
		return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return treey
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[0,tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def resultStat(predict,real):
	right = 0
#predict = round(predict)
	for i in range(len(predict)):
		if abs(round(predict[i])-real[i])<0.001:
			right +=1
	return right

def randomBagging(train,test,ntree=5,leafType=regLeaf, errType=regErr,ops=(1,4)): # a random forest approach
	d2 = shape(train)[0]
	d1 = shape(train)[1]
	print d1,d2
	train = mat(train)
	test = mat(test)
	resultMatrix = mat(zeros((len(test),3)))
	result = mat(zeros((len(test),1)))
	for t in range(ntree):
		sam = sorted(sample(range(d1-2),d1-500))
		r = [d1-1]
		trainShuffle = train[:,sam+r]
		testShuffle = test[:,sam]
		
		trainShuffle = trainShuffle[sample(range(1,d2),10),:]
		tree = createTree(trainShuffle,leafType,errType,ops)
		y = createForeCast(tree,testShuffle)
		print y
		for i in range(len(y)):
#	y[i] = round(y[i])
#		for j in range(3):
#				if y[i]==j:
#					resultMatrix[i,j]+=1
			if y[i] <= 0.66:
				resultMatrix[i,0] +=1
			elif y[i] <=1.33:
				resultMatrix[i,1] +=1
			else:
				resultMatrix[i,2] +=1
		print t
#	result = resultMatrix.argmax(axis=1)
	return resultMatrix.argmax(axis=1)[0]

# the following two functions are basically to separate the randomBagging into two so that we do not need to
# train the tree every time new data is coming.

def treeGroupForm(train,ntree=50,leafType=regLeaf, errType=regErr,ops=(1,4)):
	d2 = shape(train)[0]
	d1 = shape(train)[1]
	print d1,d2
	train = mat(train)
	trees = []
	sams = []
	for t in range(ntree):
		sam = sorted(sample(range(d1-2),d1-600))
		r = [d1-1]
		trainShuffle = train[:,sam+r]
#	testShuffle = test[:,sam]
		sams.append(sam)
		print sam
		trainShuffle = trainShuffle[sample(range(1,d2),50),:]
		tree = createTree(trainShuffle,leafType,errType,ops)
		trees.append(tree)
		print t
	return trees,sams

def batchPredict(test,trees,sams):
	resultMatrix = mat(zeros((len(test),5)))
	result = mat(zeros((len(test),1)))
	test = mat(test)
	for i in range(len(sams)):
		testShuffle = test[:,sams[i]]
		y = createForeCast(trees[i],testShuffle)
		print y
		for i in range(len(y)):
			if y[i] <= 0.5:
				resultMatrix[i,0] +=1
			elif y[i] <=0.9:
				resultMatrix[i,1] +=1
			elif y[i] <=1.1:
				resultMatrix[i,2] +=1
			elif y[i] <=1.5:
				resultMatrix[i,3] +=1
			else:
				resultMatrix[i,4] +=1
		print resultMatrix
	return resultMatrix.argmax(axis=1)[0]


def treeDecision(resultMatrix):
	return resultMatrix.argmax(axis=1)

def randomForest(train,test):
	target = [x[0] for x in train]
	trainV = [x[1:6] for x in train]
	testV = [x[0:] for x in test]

	rf = RandomForestClassifier(n_estimators=100,n_jobs=2)
	rf.fit(trainV,target)
	predicted_probs = [x[1] for x in rf.predict_proba(testV)]

	return 	predicted_probs

