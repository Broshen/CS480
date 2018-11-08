import numpy as np
from scipy import misc


def downSample1DArr(row):
	row = np.reshape(row, (28, 28))
	row = misc.imresize(row, (16, 16))
	row = np.true_divide(row, 255)
	row = np.reshape(row, (256))

	return row

def downSample():
	X = np.loadtxt(open("MNIST_Xtrain.csv", "rb"), delimiter=",")
	Xsubmit = np.loadtxt(open("MNIST_Xtestp.csv", "rb"), delimiter=",")
	smallX = np.loadtxt(open("smallX.csv", "rb"), delimiter=",")

	Xds = []
	Xsubmitds = []
	smallXds = []

	for i,row in enumerate(smallX):
		smallXds.append(downSample1DArr(row))
	smallXds = np.array(smallXds).astype(float)
	np.savetxt("smallX_downsampled.csv", smallXds, fmt="%f", delimiter=",")


	for i,row in enumerate(Xsubmit):
		Xsubmitds.append(downSample1DArr(row))
	Xsubmitds = np.array(Xsubmitds).astype(float)
	np.savetxt("MNIST_Xtestp_downsampled.csv", Xsubmitds, fmt="%f", delimiter=",")


	for i,row in enumerate(X):
		Xds.append(downSample1DArr(row))
	Xds = np.array(Xds).astype(float)
	np.savetxt("MNIST_Xtrain_downsampled.csv", Xds, fmt="%f", delimiter=",")

downSample()

