import numpy
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def loadHousing():
	Xtrain = numpy.loadtxt(open("housing_X_train.csv", "rb"), delimiter=",")
	ytrain = numpy.loadtxt(open("housing_y_train.csv", "rb"), delimiter=",")
	Xtest = numpy.loadtxt(open("housing_X_test.csv", "rb"), delimiter=",")
	ytest = numpy.loadtxt(open("housing_y_test.csv", "rb"), delimiter=",")
	Xtrain = numpy.matrix.transpose(numpy.vstack((Xtrain, [1]*len(Xtrain[0]))))
	Xtest = numpy.matrix.transpose(numpy.vstack((Xtest, [1]*len(Xtest[0]))))
	return Xtrain, ytrain, Xtest, ytest

def computeW(X, y, Lambda):
	XTX = numpy.matmul(numpy.matrix.transpose(X),X)
	LambdaI = numpy.full((len(X[0]), len(X[0])), Lambda)
	A = numpy.add(XTX, LambdaI)
	B = numpy.matmul(numpy.matrix.transpose(X),y)
	# we are working with the large array, use the swapaxes trick
	# https://stackoverflow.com/questions/48387261/numpy-linalg-solve-with-right-hand-side-of-more-than-three-dimensions
	if len(B) == 1: 
		W = numpy.swapaxes(numpy.linalg.solve(A, numpy.swapaxes(B, 0, 1)), 0, 1)
	else:
		W = numpy.linalg.solve(A,B)
	return W

def computeWLasso(X, y, Lambda, tolerance):
	allWReachedTol = False
	W = numpy.array([0]*X[0].size)

	XT = numpy.matrix.transpose(X)

	loop = 0
	while not allWReachedTol:
		#print("loop", loop)
		loop+=1
		allWReachedTol = True
		for j in range(0, len(W)):
			#print("j:", j)
			prevWj = W[j]
			v = numpy.array([0]*len(y))

			for k in range(0, len(W)):
				if j == k:
					continue
				v=numpy.add(XT[k]*W[k],v)
			# print("v1: ", v)
			# v = numpy.sum(XT*W[:, numpy.newaxis])
			# v = numpy.subtract(v, XT[j]*W[j])
			# print("v2: ", v)
			v = numpy.subtract(v,y)
			a = numpy.sum(numpy.square(XT[j]))
			b = -numpy.sum(numpy.multiply(XT[j], v))

			W[j] = numpy.sign(b/a)*max(0, abs(b/a)-2*Lambda/a)
			if abs(prevWj - W[j]) > tolerance:
				allWReachedTol = False
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

def crossValidateLasso(X, y, fold, Lambda):
	Xs = numpy.array_split(X, fold)
	ys = numpy.array_split(y, fold)
	perf = 0

	for i in range(0, fold):
		Xtest = Xs[i]
		ytest = ys[i]
		Xtrain = numpy.concatenate(numpy.delete(Xs, i))
		ytrain = numpy.concatenate(numpy.delete(ys, i))
		W = computeWLasso(Xtrain, ytrain, Lambda, 0.001)
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

def ridgeRegressionReport(Xtrain, ytrain, Xtest, ytest):

	print("Lambda\tTraining Set MSE\tAverage Validation Set MSE\tTest Set MSE\t% Nonzeros in W")

	for Lambda in range(0, 101, 10):
		score = crossValidate(Xtrain, ytrain, 10, Lambda)
		W = computeW(Xtrain, ytrain, Lambda)
		MSE_train = meanSquareError(Xtrain, W, ytrain)
		MSE_validation = score/10
		MSE_test = meanSquareError(Xtest, W, ytest)
		# numpy turns W into a matrix instead of array when doing large matrix operations (i.e. > 1000 rows)
		if isinstance(W[0], numpy.matrixlib.defmatrix.matrix):
			nonzeros = numpy.count_nonzero(W)*100/len(W.getA1())
		else:
			nonzeros = numpy.count_nonzero(W)*100/len(W)
		print("{}\t{}\t\t{}\t\t\t{}\t{}".format(Lambda, round(MSE_train,10), round(MSE_validation,10), round(MSE_test,10), nonzeros))

def lassoRegressionReport(Xtrain, ytrain, Xtest, ytest):

	print("Lambda\tTraining Set MSE\tAverage Validation Set MSE\tTest Set MSE\t% Nonzeros in W")

	for Lambda in range(0, 101, 10):
		score = crossValidateLasso(Xtrain, ytrain, 10, Lambda)
		W = computeWLasso(Xtrain, ytrain, Lambda, 0.001)
		MSE_train = meanSquareError(Xtrain, W, ytrain)
		MSE_validation = score/10
		MSE_test = meanSquareError(Xtest, W, ytest)
		# numpy turns W into a matrix instead of array when doing large matrix operations (i.e. > 1000 rows)
		if isinstance(W[0], numpy.matrixlib.defmatrix.matrix):
			nonzeros = numpy.count_nonzero(W)*100/len(W.getA1())
		else:
			nonzeros = numpy.count_nonzero(W)*100/len(W)
		print("{}\t{}\t\t{}\t\t\t{}\t{}".format(Lambda, round(MSE_train,10), round(MSE_validation,10), round(MSE_test,10), nonzeros))

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

def Exercise2P4():
	Xtrain, ytrain, Xtest, ytest = loadHousing()
	lassoRegressionReport(Xtrain, ytrain, Xtest, ytest)

def Exercise2P5():
	Xtrain, ytrain, Xtest, ytest = loadHousing()
	Xtrain, Xtest = addRandomFeatures(1000, Xtrain, Xtest)
	lassoRegressionReport(Xtrain, ytrain, Xtest, ytest)

# Exercise2P1()
# Exercise2P2()
# Exercise2P3()
Exercise2P4()
Exercise2P5()



