import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter

sampleX = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.21569,0.53333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.67451,0.99216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.070588,0.88627,0.99216,0,0,0,0,0,0,0,0,0,0,0.19216,0.070588,0,0,0,0,0,0,0,0,0,0,0,0,0,0.67059,0.99216,0.99216,0,0,0,0,0,0,0,0,0,0.11765,0.93333,0.85882,0.31373,0,0,0,0,0,0,0,0,0,0,0,0.090196,0.85882,0.99216,0.83137,0,0,0,0,0,0,0,0,0,0.14118,0.99216,0.99216,0.61176,0.054902,0,0,0,0,0,0,0,0,0,0,0.25882,0.99216,0.99216,0.52941,0,0,0,0,0,0,0,0,0,0.36863,0.99216,0.99216,0.41961,0.0039216,0,0,0,0,0,0,0,0,0,0.094118,0.83529,0.99216,0.99216,0.51765,0,0,0,0,0,0,0,0,0,0.60392,0.99216,0.99216,0.99216,0.60392,0.5451,0.043137,0,0,0,0,0,0,0,0.44706,0.99216,0.99216,0.95686,0.062745,0,0,0,0,0,0,0,0,0.011765,0.66667,0.99216,0.99216,0.99216,0.99216,0.99216,0.7451,0.13725,0,0,0,0,0,0.15294,0.86667,0.99216,0.99216,0.52157,0,0,0,0,0,0,0,0,0,0.070588,0.99216,0.99216,0.99216,0.80392,0.35294,0.7451,0.99216,0.9451,0.31765,0,0,0,0,0.58039,0.99216,0.99216,0.76471,0.043137,0,0,0,0,0,0,0,0,0,0.070588,0.99216,0.99216,0.77647,0.043137,0,0.0078431,0.27451,0.88235,0.94118,0.17647,0,0,0.18039,0.89804,0.99216,0.99216,0.31373,0,0,0,0,0,0,0,0,0,0,0.070588,0.99216,0.99216,0.71373,0,0,0,0,0.62745,0.99216,0.72941,0.062745,0,0.5098,0.99216,0.99216,0.77647,0.035294,0,0,0,0,0,0,0,0,0,0,0.49412,0.99216,0.99216,0.96863,0.16863,0,0,0,0.42353,0.99216,0.99216,0.36471,0,0.71765,0.99216,0.99216,0.31765,0,0,0,0,0,0,0,0,0,0,0,0.53333,0.99216,0.98431,0.9451,0.60392,0,0,0,0.0039216,0.46667,0.99216,0.98824,0.97647,0.99216,0.99216,0.78824,0.0078431,0,0,0,0,0,0,0,0,0,0,0,0.68627,0.88235,0.36471,0,0,0,0,0,0,0.098039,0.58824,0.99216,0.99216,0.99216,0.98039,0.30588,0,0,0,0,0,0,0,0,0,0,0,0,0.10196,0.67451,0.32157,0,0,0,0,0,0,0,0.10588,0.73333,0.97647,0.81176,0.71373,0,0,0,0,0,0,0,0,0,0,0,0,0,0.65098,0.99216,0.32157,0,0,0,0,0,0,0,0,0,0.25098,0.0078431,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.94902,0.21961,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.96863,0.76471,0.15294,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.49804,0.25098,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
sampleY = [5]

sampleX = np.array(sampleX)
sampleY = np.array(sampleY)


def moments(image):
	c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
	totalImage = np.sum(image) #sum of pixels
	m0 = np.sum(c0*image)/totalImage #mu_x
	m1 = np.sum(c1*image)/totalImage #mu_y
	m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
	m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
	m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
	mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
	covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
	return mu_vector, covariance_matrix

def deskew(image):
	image = image.reshape(28,28)
	c,v = moments(image)
	alpha = v[0,1]/v[0,0]
	affine = np.array([[1,0],[alpha,1]])
	ocenter = np.array(image.shape)/2.0
	offset = c-np.dot(affine,ocenter)
	image = interpolation.affine_transform(image,affine,offset=offset)
	image = (image - image.min()) / (image.max() - image.min())
	return image.reshape(784)


def deskew_all():
	X = np.loadtxt(open("MNIST_Xtrain.csv", "rb"), delimiter=",")
	Xsubmit = np.loadtxt(open("MNIST_Xtestp.csv", "rb"), delimiter=",")
	smallX = np.loadtxt(open("smallX.csv", "rb"), delimiter=",")

	Xds = []
	Xsubmitds = []
	smallXds = []

	for i,row in enumerate(smallX):
		smallXds.append(deskew(row))
	smallXds = np.array(smallXds).astype(float)
	np.savetxt("smallX_deskewed_normalized.csv", smallXds, fmt="%f", delimiter=",")


	for i,row in enumerate(Xsubmit):
		Xsubmitds.append(deskew(row))
	Xsubmitds = np.array(Xsubmitds).astype(float)
	np.savetxt("MNIST_Xtestp_deskewed_normalized.csv", Xsubmitds, fmt="%f", delimiter=",")


	for i,row in enumerate(X):
		Xds.append(deskew(row))
	Xds = np.array(Xds).astype(float)
	np.savetxt("MNIST_Xtrain_deskewed_normalized.csv", Xds, fmt="%f", delimiter=",")

def gaussian_filter_all():
	X = np.loadtxt(open("MNIST_Xtrain.csv", "rb"), delimiter=",")
	Xsubmit = np.loadtxt(open("MNIST_Xtestp.csv", "rb"), delimiter=",")
	smallX = np.loadtxt(open("smallX.csv", "rb"), delimiter=",")

	Xds = []
	Xsubmitds = []
	smallXds = []

	for i,row in enumerate(smallX):
		smallXds.append(gaussian_filter(row.reshape(28,28), 1).reshape(784))
	smallXds = np.array(smallXds).astype(float)
	np.savetxt("smallX_blurred.csv", smallXds, fmt="%f", delimiter=",")


	for i,row in enumerate(Xsubmit):
		Xsubmitds.append(gaussian_filter(row.reshape(28,28), 1).reshape(784))
	Xsubmitds = np.array(Xsubmitds).astype(float)
	np.savetxt("MNIST_Xtestp_blurred.csv", Xsubmitds, fmt="%f", delimiter=",")


	for i,row in enumerate(X):
		Xds.append(gaussian_filter(row.reshape(28,28), 1).reshape(784))
	Xds = np.array(Xds).astype(float)
	np.savetxt("MNIST_Xtrain_blurred.csv", Xds, fmt="%f", delimiter=",")


deskew_all()

