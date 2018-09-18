import numpy
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def loadHousing():
	Xtrain = numpy.matrix.transpose(numpy.loadtxt(open("housing_X_train.csv", "rb"), delimiter=","))
	ytrain = numpy.loadtxt(open("housing_y_train.csv", "rb"), delimiter=",")
	Xtest = numpy.matrix.transpose(numpy.loadtxt(open("housing_X_test.csv", "rb"), delimiter=","))
	ytest = numpy.loadtxt(open("housing_y_test.csv", "rb"), delimiter=",")
	return Xtrain, ytrain, Xtest, ytest

def loadTestSet():
	Xtrain = numpy.loadtxt(open("X_train.csv", "rb"), delimiter=",")
	ytrain = numpy.loadtxt(open("y_train.csv", "rb"), delimiter=",")
	Xtest = numpy.loadtxt(open("X_test.csv", "rb"), delimiter=",")
	ytest = numpy.loadtxt(open("y_test.csv", "rb"), delimiter=",")
	return Xtrain, ytrain, Xtest, ytest

def computeW(X, y, Lambda):
	XTX = numpy.matmul(numpy.matrix.transpose(X),X)
	LambdaI = numpy.full((len(X[0]), len(X[0])), Lambda)
	A = numpy.add(XTX, LambdaI)
	B = numpy.matmul(numpy.matrix.transpose(X),y)
	W = numpy.linalg.solve(A,B) #.reshape(B.shape[0], -1)).reshape(B.shape)#)
	#W = numpy.swapaxes(numpy.linalg.solve(A, numpy.swapaxes(B, 0, 1)), 0, 1)
	return W

def meanSquareError(X, W, y):
	return numpy.sum(numpy.square(numpy.subtract(numpy.matmul(X,numpy.matrix.transpose(W)),y)))/len(y)

# given a lambda, return it's score
def crossValidate(X, y, fold, Lambda):
	Xs = numpy.array_split(X, fold)
	ys = numpy.array_split(y, fold)
	perf = 0

	for i in range(0, fold):
		Xtest = Xs[i]
		ytest = ys[i]
		Xtrain = numpy.concatenate(numpy.delete(Xs, i))
		ytrain = numpy.concatenate(numpy.delete(ys, i))
		W = computeW(Xtrain, ytrain, Lambda)
		perf += meanSquareError(Xtest, W, ytest)

	return perf

def addRandomFeatures(numToAdd, Xtrain, Xtest):
	Xtrain = numpy.matrix.transpose(Xtrain)
	Xtest = numpy.matrix.transpose(Xtest)
	l1 = len(Xtrain[0])
	l2= len(Xtest[0])
	trainRandom = []
	testRandom = []


	for num in range(numToAdd):
		trainRandom.append(numpy.random.standard_normal(l1))
		testRandom.append(numpy.random.standard_normal(l2))

	trainRandom = numpy.matrix(trainRandom)
	testRandom = numpy.matrix(testRandom)

	Xtrain = numpy.matrix.transpose(numpy.concatenate([Xtrain, trainRandom]))
	Xtest = numpy.matrix.transpose(numpy.concatenate([Xtest, testRandom]))

	return Xtrain, Xtest

# def findLambdas(minL, maxL, increment, X, y):
# 	scores = []
# 	lambdas = []
# 	for candidate in range(minL, maxL+increment, increment):
# 		score = crossValidate(X, y, 10, candidate)
# 		lambdas.append(candidate)
# 		scores.append(score)

# 	return lambdas[numpy.argmin(scores)], lambdas, scores

# def ridgeRegression(Xtrain, ytrain, Xtest, ytest):
# 	Lambdas = findLambdas(0, 100, 10, Xtrain, ytrain)
# 	W = computeW(Xtrain, ytrain, Lambda[0])
# 	return meanSquareError(Xtest, W, ytest)

def ridgeRegressionReport(Xtrain, ytrain, Xtest, ytest):

	print "Lambda\tTraining Set MSE\tAverage Validation Set MSE\tTest Set MSE\t% Nonzeros in W"

	for Lambda in range(0, 101, 10):
		score = crossValidate(Xtrain, ytrain, 10, Lambda)
		W = computeW(Xtrain, ytrain, Lambda)
		MSE_train = meanSquareError(Xtrain, W, ytrain)
		MSE_validation = score/10
		MSE_test = meanSquareError(Xtest, W, ytest)
		nonzeros = numpy.count_nonzero(W)/len(W)*100
		print "{}\t{}\t\t{}\t\t\t{}\t{}".format(Lambda, round(MSE_train,10), round(MSE_validation,10), round(MSE_test,10), nonzeros)

def Exercise2P1():
	Xtrain, ytrain, Xtest, ytest = loadHousing()
	ridgeRegressionReport(Xtrain, ytrain, Xtest, ytest)

def Exercise2P2():
	Xtrain, ytrain, Xtest, ytest = loadHousing()
	i = numpy.random.randint(0, len(ytrain))
	Xtrain[i] = numpy.multiply(Xtrain[i], 1000000)
	ytrain[i] = ytrain[i] * 1000
	ridgeRegressionReport(Xtrain, ytrain, Xtest, ytest)

def Exercise2P3():
	Xtrain, ytrain, Xtest, ytest = loadHousing()
	Xtrain, Xtest = addRandomFeatures(1000, Xtrain, Xtest)
	ridgeRegressionReport(Xtrain, ytrain, Xtest, ytest)


def QA():
	Xtrain, ytrain, Xtest, ytest = loadTestSet()
	W = computeW(Xtrain, ytrain, 0)
	meanSquareError(Xtest, W, ytest)
	addRandomFeatures(10, Xtrain, Xtest)

#QA()

Exercise2P1()
Exercise2P2()
#Exercise2P3()



