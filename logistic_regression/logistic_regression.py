import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

def write2csv(data, filename = "out.csv"):
	y = pd.Series(data, index = range(1, len(data) + 1) , name = "Prediction")
	y.to_csv(filename, header=True, index_label="Id")

def readTrainData(filename = "../forest_train.csv"):
	train = pd.read_csv(filename, header = None)
	return (train.values[:, 0:-1], train.values[:, -1])

def readTestData(filename = "../forest_test.csv"):
	test = pd.read_csv(filename, header = None)
	return test.values[:,:]

def  main():
	(X, Y) = readTrainData()
	
	# this is my randomized search cross validation
	logreg = linear_model.LogisticRegression()
	
	logreg.fit(X, Y)
	param_grid = {
              "C": [1.0, 10., 100.],
			  'penalty':["l1", "l2"],
	}
	rand_search = RandomizedSearchCV(logreg, n_iter=5, param_distributions=param_grid)
	rand_search.fit(X, Y)
	best_parameters = rand_search.best_params_
	print best_parameters

	#train my classifiers with the parameters from cross validation
	logreg = linear_model.LogisticRegression(penalty = best_parameters['penalty'], \
		C = best_parameters['C'])
	
	#the following parameters are for reference
	#logreg = linear_model.LogisticRegression(penalty = 'l1', C = 10.0)
	
	logreg.fit(X, Y)
	(validateX, validateY) = readTrainData("../forest_validation.csv") 
	predictY = logreg.predict(validateX)
	
	resultfile = open('confusionmatrix_test.txt','w')
	resultfile.write(str(confusion_matrix(validateY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(validateY, predictY))
	resultfile.close()	
	print 'accuracy: %.4f' % ( ( sum( predictY == validateY ) / float(len(predictY)))) 

	testX = readTestData()
	testY = logreg.predict(testX)
	write2csv(testY, filename = "output_logisistic_regression.csv")



if __name__ == "__main__":
	main()

