import numpy as np
from sklearn.preprocessing import normalize
import math
import time
import datetime

def loadData():
	X_train = np.loadtxt(open("train_X_dog_cat.csv", "rb"), delimiter=",")
	X_test = np.loadtxt(open("test_X_dog_cat.csv", "rb"), delimiter=",")
	y_train = np.loadtxt(open("train_y_dog_cat.csv", "rb"), delimiter=",")
	y_test = np.loadtxt(open("test_y_dog_cat.csv", "rb"), delimiter=",")

	return np.divide(X_train, 255), np.divide(X_test, 255), y_train, y_test

# X: training data matrix - n*d
# k: kernel function - f(list, list)
# returns: n*n matrix
def getKernelMatrix(X, k):
	K = np.matrix([[0]*len(X)]*len(X))

	for i in range(0, len(X)):
		for j in range(0, len(X)):
			K[i,j] = k(X[i], X[j])

	return normalize(K)

def KLRSGD(X, y, K, Lambda, maxIter=10000 , stepSize=0.0001, tol=0.01):
	n = len(X)
	alpha = np.array([0]*n)
	
	for Iter in range(0, maxIter):
	

		# 2*Lambda*alpha*K
		regularization = np.multiply(2*Lambda, np.matmul(K, np.multiply(y,alpha)))

		# p = 1/(1+exp(-alpha.T * K))
		p = 1/(1+ np.exp(np.matmul(-alpha.T,K)))

		# g = K.T * (p - (y + 1)/2) + 2*Lambda*alpha*K
		g = np.matmul(K.T, (p - np.add(y, 1)/2)) + regularization

		# alpha = alpha - n*g
		alpha = np.subtract(alpha, np.multiply(stepSize, g))

		# debugging - uncomment to see
		# if Iter%(maxIter/5) == 0: #and i%(n/3) == 0:
		# 	print("step size: ")
		# 	print(np.multiply(stepSize, g), (g > 0).sum(), (g <= 0).sum())
		# 	print("alpha: ")
		# 	print(alpha, (alpha > 0).sum(), (alpha <= 0).sum())
		# 	print("logistic loss:")
		# 	print(np.log(1/p))
		# 	print("\n")

	return alpha

def predict(X_train, x_test, y_train, alpha, k):
	wTx = 0

	# our prediction is
	# wTx = sum(alpha_i * k(x_i, x_test))
	for xi, ai in zip(X_train, alpha):
		wTx += ai*k(xi, x_test)

	return wTx

def score(X_train, X_test, y_train, y_test, alpha, k):
	preds = []
	correct = 0

	for i, yi in enumerate(y_test):
		pred = predict(X_train, X_test[i], y_train, alpha, k)
		preds.append(pred)

		if pred*yi >= 0:
			correct += 1.0

	print("Accuracy: " + str(correct/len(y_test)))


def testKernel(X_train, X_test, y_train, y_test, k, Lambda):
	K = getKernelMatrix(X_train, k)
	alpha = KLRSGD(X_train, y_train, K, Lambda)
	score(X_train, X_test, y_train, y_test, alpha, k)







def linearKernel(x1, x2):
	return np.dot(x1, x2)

def IPkernel(x1, x2):
	return pow(1+np.dot(x1, x2), 5)

def makeGaussianKernel(sigma):
	def gaussianKernel(x1, x2):
		dx = np.subtract(x1, x2)
		dx = -1 * pow(np.linalg.norm(dx), 2)/sigma
		return math.exp(dx)

	return gaussianKernel






X_train, X_test, y_train, y_test = loadData()
lambdas = [0,10,20,30,40,50,60,70,80,90,100]

print("linear kernel")
for l in lambdas:
	print("lambda = " + str(l))
	testKernel(X_train, X_test, y_train, y_test, linearKernel, l)

print("inhomogeneous polynomial kernel")
for l in lambdas:
	print("lambda = " + str(l))
	testKernel(X_train, X_test, y_train, y_test, IPkernel, l)

print("gaussian kernel")
sigmas = [1,2,3,4,5,6,7,8,9,10]
for l in lambdas:
	print("lambda = " + str(l))
	for s in sigmas:
		print("sigma = " + str(s))
		testKernel(X_train, X_test, y_train, y_test, makeGaussianKernel(s), l)