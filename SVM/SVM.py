import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn import svm

def write2csv(data, filename = "out_logistic.csv"):
	y = pd.Series(data, index = range(1, len(data) + 1) , name = "Prediction")
	y.to_csv(filename, header=True, index_label="Id")

def readTrainData(filename = "../forest_train.csv"):
	train = pd.read_csv(filename, header = None)
	return (train.values[:, 0:-1], train.values[:, -1])

def normalizeData(dat, scalars = None):
	dataset = dat
	if not scalars:           
		scalars = preprocessing.StandardScaler().fit(X=dataset.astype(np.float))
	datset = scalars.transform(X=dataset.astype(np.float))
	return (datset, scalars)  

def readTestData(filename = "../forest_test.csv"):
	train = pd.read_csv(filename, header = None)
	return train.values[:, :]


def  main():
	(trainX, trainY) = readTrainData()
	(first10columnX, scalars) = normalizeData(trainX[:, 0:10])
	clf = OneVsRestClassifier(svm.SVC(max_iter=16000))
	clfbest = clf.set_params(estimator__kernel='rbf',estimator__gamma=1.0,estimator__C=2.0) 
	trainX = np.concatenate((first10columnX[:,:], trainX[:,10:50]), axis = 1)
	clfbest.fit(trainX, trainY)
	
	'''
	the cross validation can be done manually but extremely time consuming
	on my 2013 Mac, it will take at least 90 min
	the parameters are chosen but a cross validation with much smaller input data.
	'''


	(validateX, validateY) = readTrainData("../forest_validation.csv") 
	first10validateX = normalizeData(validateX[:, 0:10], scalars)[0]
	validateX = np.concatenate((first10validateX[:,:], validateX[:,10:50]), axis = 1)
	predictY = clfbest.predict(validateX)
	resultfile = open('confusionmatrix_test.txt','w')
	resultfile.write(str(confusion_matrix(validateY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(validateY, predictY))
	resultfile.close()	
	
	print 'accuracy: %.4f' % ( ( sum( predictY == validateY ) / float(len(predictY)))) 

	testX = readTestData()
	first10testX = normalizeData(testX[:, 0:10], scalars)[0]
	testX = np.concatenate((first10testX[:,:], testX[:,10:50]), axis = 1)
	testY = clfbest.predict(testX)
	write2csv(testY, filename = "output_SVM.csv")

if __name__ == "__main__":
	main()
