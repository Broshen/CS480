import numpy as np
from sklearn import neighbors, svm, linear_model, tree, ensemble, base
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import time
import datetime

def loadAndSplitData(Xfile, yfile):
	print("loading data")

	X = np.loadtxt(open(Xfile, "rb"), delimiter=",")
	y = np.loadtxt(open(yfile, "rb"), delimiter=",")
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

	return X, Xtrain, Xtest, y, ytrain, ytest

def testAllClassifiers(Xfile, yfile):
	X, Xtrain, Xtest, y, ytrain, ytest = loadAndSplitData(Xfile, yfile)
	clfs = [
		linear_model.Perceptron(max_iter=1000),
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
		"kNN, k=15",
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


def testAllKNeighbors(Xfile, yfile, lo, hi, step):
	X, Xtrain, Xtest, y, ytrain, ytest = loadAndSplitData(Xfile, yfile)
	for i in range(lo, hi, step):
		clf1 = neighbors.KNeighborsClassifier(i, weights='uniform')
		clf1.fit(Xtrain, ytrain)
		print("KNN, uniform, k="+str(i)+":",clf1.score(Xtest,ytest))
		clf2 = neighbors.KNeighborsClassifier(i, weights='distance')
		clf2.fit(Xtrain, ytrain)
		print("KNN, distance, k="+str(i)+":",clf2.score(Xtest,ytest))

def testClassifier(Xfile, yfile, classifier, label):
	X, Xtrain, Xtest, y, ytrain, ytest = loadAndSplitData(Xfile, yfile)
	ts = time.time()
	print("fitting " + label)
	classifier.fit(Xtrain, ytrain)
	ts = (time.time() - ts)
	print(label+'\t', classifier.score(Xtest, ytest),
		"\ttime taken: " + str(datetime.timedelta(seconds=int(ts))) + "\t num rows: " + str(len(ytest) + len(ytrain)))


def testGaussianFilters(Xfile, yfile, classifiers, classifierNames, lambdas):
	X, Xtrain, Xtest, y, ytrain, ytest = loadAndSplitData(Xfile, yfile)

	for i,clf in enumerate(classifiers):
		for l in lambdas:
			Xtrain_gauss = []
			Xtest_gauss = []
			classifier = base.clone(clf)
			for row in Xtrain:
				Xtrain_gauss.append(gaussian_filter(row.reshape(28,28), l).reshape(784))
			for row in Xtest:
				Xtest_gauss.append(gaussian_filter(row.reshape(28,28), l).reshape(784))
			classifier.fit(Xtrain_gauss, ytrain)
			print("lambda:",l,"\t",classifierNames[i],"score:",classifier.score(Xtest_gauss,ytest))



def ceilAllData(Xfiles):
	for file in Xfiles:
		X = np.loadtxt(open(file+".csv", "rb"), delimiter=",")
		X = np.ceil(X).astype(int)
		np.savetxt(file+"_BW.csv", X, fmt="%i", delimiter=",")


def createSubmission(Xfile, yfile, submissionXFile, classifier, submissionFileName, traintestFirst=False):
	print("loading submission data")
	X = np.loadtxt(open(Xfile, "rb"), delimiter=",")
	y = np.loadtxt(open(yfile, "rb"), delimiter=",")
	Xsubmit = np.loadtxt(open(submissionXFile, "rb"), delimiter=",")

	if traintestFirst:
		Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=0)
		print("training on train/test split")
		classifier.fit(Xtrain, ytrain)
		print("train/test split score: ", classifier.score(Xtest, ytest))

	print("fitting to full test data")
	classifier = base.clone(classifier)
	classifier.fit(X,y)

	print("making predictions")
	ysubmit = classifier.predict(Xsubmit)

	print("writing submissions to " + submissionFileName)
	with open(submissionFileName, "w+") as submission:
		submission.write("ImageID,Digit\n")
		for i,y in enumerate(ysubmit):
			submission.write(str(int(i+1))+','+str(int(y))+'\n')

createSubmission(
	"MNIST_Xtrain_deskewed_normalized.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp_deskewed_normalized.csv",
	neighbors.KNeighborsClassifier(6, weights='distance'),
	"KNN6_deskewed_normalized.csv"
)
createSubmission(
	"MNIST_Xtrain.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp.csv",
	ensemble.BaggingClassifier(
		base_estimator=neighbors.KNeighborsClassifier(6, weights='distance'),
		n_estimators=50,
	),
	"BaggingKNN.csv"
)
createSubmission(
	"MNIST_Xtrain_deskewed.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp_deskewed.csv",
	neighbors.KNeighborsClassifier(6, weights='distance'),
	"KNN6_deskewed.csv"
)
createSubmission(
	"MNIST_Xtrain.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp.csv",
	neighbors.KNeighborsClassifier(6, weights='distance'),
	"KNN6.csv"
)
createSubmission(
	"MNIST_Xtrain.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp.csv",
	neighbors.KNeighborsClassifier(6, weights='distance', metric="minkowski", metric_params={'p':3}),
	"KNN6_L3.csv"
)
createSubmission(
	"MNIST_Xtrain_blurred.csv",
	"MNIST_ytrain.csv",
	"MNIST_Xtestp_blurred.csv",
	neighbors.KNeighborsClassifier(6, weights='distance', metric="minkowski", metric_params={'p':3}),
	"KNN6_L3_gaussian_blurred.csv"
)
# testClassifier(
# 	"smallX.csv",
# 	"smally.csv",
# 	neighbors.KNeighborsClassifier(6, weights='distance', metric="minkowski", metric_params={'p':3}),
# 	"small dataset, kNN, k=6, deskewed, normalized, L3norm"
# )

# testClassifier(
# 	"smallX.csv",
# 	"smally.csv",
# 	ensemble.AdaBoostClassifier(
# 		base_estimator=tree.DecisionTreeClassifier(max_leaf_nodes=17),
# 		n_estimators=1000
# 	),
# 	"small dataset, adaboosted trees, n=1000, leaves=17"
# )

# testClassifier(
# 	"smallX.csv",
# 	"smally.csv",
# 	ensemble.RandomForestClassifier(
# 		max_leaf_nodes=17,
# 		n_estimators=1000
# 	),
# 	"small dataset, randomforest, n=1000, leaves=17"
# )

# testClassifier(
# 	"smallX_deskewed.csv",
# 	"smally.csv",
# 	ensemble.AdaBoostClassifier(
# 		base_estimator=tree.DecisionTreeClassifier(max_leaf_nodes=17),
# 		n_estimators=1000
# 	),
# 	"small dataset, deskewed, adaboosted trees, n=1000, leaves=17"
# )

# testClassifier(
# 	"smallX_deskewed.csv",
# 	"smally.csv",
# 	ensemble.RandomForestClassifier(
# 		max_leaf_nodes=17,
# 		n_estimators=1000
# 	),
# 	"small dataset, deskewed, randomforest, n=1000, leaves=17"
# )

# testAllKNeighbors(
# 	"smallX_deskewed.csv",
# 	"smally.csv",
# 	1,
# 	25,
# 	1
# )



