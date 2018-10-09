import numpy as np
from sklearn import neighbors, svm, linear_model, tree, ensemble, base
from sklearn.model_selection import train_test_split

def testAllClassifiers(Xtrain, Xtest, ytrain, ytest):
	clfs = [
		linear_model.Perceptron(max_iter=10000),
		neighbors.KNeighborsClassifier(15, weights='uniform'),		
		linear_model.LogisticRegression(),
		tree.DecisionTreeClassifier(),
		ensemble.BaggingClassifier(),
		ensemble.AdaBoostClassifier(),
		ensemble.RandomForestClassifier(),
		svm.LinearSVC()
	]

	clfNames = [
		"perceptron",
		"kNN",
		"logistic regression",
		"decision tree",
		"bagging",
		"boosting",
		"random forest",
		"support vector machines"
	]

	for i, clf in enumerate(clfs):
		clf.fit(Xtrain, ytrain)
		print(clfNames[i] + " :", clf.score(Xtest, ytest))


def testAllKNeighbors(Xtrain, Xtest, ytrain, ytest, lo, hi, step):
	for i in range(lo, hi, step):
		clf1 = neighbors.KNeighborsClassifier(i, weights='uniform')
		clf1.fit(Xtrain, ytrain)
		print("KNN, uniform, k="+str(i)+":",clf1.score(Xtest,ytest))
		clf2 = neighbors.KNeighborsClassifier(i, weights='distance')
		clf2.fit(Xtrain, ytrain)
		print("KNN, distance, k="+str(i)+":",clf2.score(Xtest,ytest))


def testPerceptronIters(Xtrain, Xtest, ytrain, ytest, lo, hi, step):
	for i in range(lo, hi, step):
		clf = linear_model.Perceptron(max_iter=i)
		clf.fit(Xtrain, ytrain)
		print("Perceptron, max_iter="+str(i)+":",clf.score(Xtest,ytest))

def testEnsembleClassifiers(Xtrain, Xtest, ytrain, ytest):
	print("starting ensemble test")
	n=50
	clf1 = ensemble.BaggingClassifier(base_estimator=neighbors.KNeighborsClassifier(6, weights='distance'), n_estimators=50)
	clf2 = ensemble.AdaBoostClassifier(base_estimator=svm.LinearSVC(), n_estimators=50, algorithm='SAMME')
	clf1.fit(Xtrain, ytrain)
	print("Bagging kNN, k=6, n="+str(n)+":",clf1.score(Xtest,ytest))
	clf2.fit(Xtrain, ytrain)
	print("AdaBoost, SVC, n=n="+str(n)+":",clf2.score(Xtest,ytest))

def createSubmission(classifier, submissionFileName):
	print("loading submission data")
	X = np.loadtxt(open("MNIST_Xtrain.csv", "rb"), delimiter=",")
	y = np.loadtxt(open("MNIST_ytrain.csv", "rb"), delimiter=",")
	Xsubmit = np.loadtxt(open("MNIST_Xtestp.csv", "rb"), delimiter=",")
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=0)

	print("training on train/test split")
	classifier.fit(Xtrain, ytrain)
	print("train/test split score: ", classifier.score(Xtest, ytest))

	print("fitting to full test data")
	classifier = base.clone(classifier)
	classifier.fit(X,y)

	
	print("making predictions")
	ysubmit = clf.predict(Xsubmit)

	print("writing submissions to " + submissionFileName)
	with open(submissionFileName, "w+") as submission:
		submission.write("ImageID,Digit\n")
		for i,y in enumerate(ysubmit):
			submission.write(str(int(i))+','+str(int(y))+'\n')


print("loading data")

X = np.loadtxt(open("smallX.csv", "rb"), delimiter=",")
y = np.loadtxt(open("smally.csv", "rb"), delimiter=",")
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# testAllClassifiers(Xtrain, Xtest, ytrain, ytest)
# testAllKNeighbors(Xtrain, Xtest, ytrain, ytest, 1, 25, 1)
# testPerceptronIters(Xtrain, Xtest, ytrain, ytest, 1000,101000,5000)
# testEnsembleClassifiers(Xtrain, Xtest, ytrain, ytest)

createSubmission(ensemble.BaggingClassifier(base_estimator=neighbors.KNeighborsClassifier(6, weights='distance'), n_estimators=50), "baggingKNN.csv")