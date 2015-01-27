import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.stats import randint as sp_randint

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
	test = pd.read_csv(filename, header = None)
	return test.values[:,:]

def main():
	(trainX, trainY) = readTrainData()
	(first10columnX, scalars) = normalizeData(trainX[:,0:10])
	trainX = np.concatenate((first10columnX[:,:], trainX[:,10:50]), axis = 1)
	
		
	#this is cross validation
	clf = RandomForestClassifier () # n_estimators = 35 
	clf = clf.fit(trainX, trainY)
	param_grid = {
			  "n_estimators" : sp_randint(30, 40),
              "max_features": sp_randint(10,20),
			  "min_samples_leaf": sp_randint(1,5),
              "min_samples_split": sp_randint(1,5),
			  "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
			  }

	grid_search = RandomizedSearchCV(clf, n_iter=20, param_distributions=param_grid)
	grid_search.fit(trainX, trainY)
	best_parameters = grid_search.best_params_
	
	#the following parameters are good 
	#best_parameters = {'n_estimators' : 35, 'max_features' : 17, 'min_samples_split' : 3, \
	#		'min_samples_leaf' : 3, 'bootstrap' : False, 'criterion': "entropy"}
	
	print best_parameters

	clf = RandomForestClassifier ( n_estimators = best_parameters['n_estimators'], 
		max_features = best_parameters['max_features'], 
		min_samples_leaf = best_parameters['min_samples_leaf'], 
		min_samples_split = best_parameters['min_samples_split'], 
		bootstrap = best_parameters['bootstrap'],
		criterion = best_parameters['criterion'] )

	clf.fit(trainX, trainY)
	(validateX, validateY) = readTrainData("../forest_validation.csv") 
	first10validateX = normalizeData(validateX[:, 0:10], scalars)[0]
	validateX = np.concatenate((first10validateX[:,:], validateX[:,10:50]), axis = 1)
	predictY = clf.predict(validateX)
	resultfile = open('confusionmatrix_test.txt','w')
	resultfile.write(str(confusion_matrix(validateY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(validateY, predictY))
	resultfile.close()	
	
	print 'accuracy: %.4f' % ( ( sum( predictY == validateY ) / float(len(predictY)))) 
	
	testX = readTestData()
	first10testX = normalizeData(testX[:, 0:10], scalars)[0]
	testX = np.concatenate((first10testX[:,:], testX[:,10:50]), axis = 1)
	testY = clf.predict(testX)
	write2csv(testY, filename = "output_random_forest.csv")

if __name__ == "__main__":
	main()
