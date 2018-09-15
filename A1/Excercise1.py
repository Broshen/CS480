import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def plotGraph(data, xlabel, ylabel, title, filename, ymin=-1, ymax=-1):
	plt.plot(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if ymin != -1:
		plt.ylim(ymin=ymin)
	if ymax != -1:
		plt.ylim(ymax=ymax)
	plt.savefig(filename)
	plt.clf()

def Perceptron(X, y, w, b, max_pass=500):
	mistake= [0]*max_pass
	n = len(y)

	for passes in range(0, max_pass):
		mistake[passes] = 0
		for i in range(0, n):
			if y[i]*(numpy.dot(X[i],w)+b) <= 0:
				w = numpy.add(w, numpy.multiply(y[i],X[i]))
				b += y[i]
				mistake[passes]+=1
	return w,b,mistake

def FlawedPerceptron(X, y, w, b, max_pass=500):
	mistake= [0]*max_pass
	n = len(y)

	X, y = shuffle(X,y)
	for passes in range(0, max_pass):
		mistake[passes] = 0
		for i in range(0, n):
			if y[i]*(numpy.dot(X[i],w)+b) <= 0:
				mistake[passes]+=1
			w = numpy.add(w, numpy.multiply(y[i],X[i]))
			b += y[i]

	return w,b,mistake


def Exercise1P1(X, y, w, b, max_pass=500):
	w,b,mistakes = Perceptron(X, y, w, b, max_pass)
	plotGraph(
		mistakes,
		xlabel="Passes",
		ylabel="Mistakes",
		title="Exercise 1 Q1",
		filename="Exercise1Q1.png"
	)

def Exercise1P2(X, y, w, b, max_pass=500):
	w,b,mistakes = FlawedPerceptron(X, y, w, b, max_pass)
	plotGraph(
		mistakes,
		xlabel="Passes",
		ylabel="Mistakes",
		title="Exercise 1 Q2",
		filename="Exercise1Q2.png",
		ymin=0,
		ymax=max(mistakes)*1.1
	)

def Exercise1P4(X, y, w,b, max_pass=500):
	for i in range(0, 5):
		Xi, yi = shuffle(X,y)
		w1,b1,mistakes = Perceptron(Xi, yi, w, b, max_pass)
		plotGraph(mistakes,
			xlabel="Passes",
			ylabel="Mistakes",
			title="Exercise 1 Q4 Plot {}".format(i),
			filename="Exercise1Q4-Plot{}.png".format(i)
		)

def loadSpambase():
	X = numpy.matrix.transpose(numpy.loadtxt(open("spambase_X.csv", "rb"), delimiter=","))
	y = numpy.loadtxt(open("spambase_y.csv", "rb"), delimiter=",")
	return X,y

X,y = loadSpambase()
w = numpy.zeros(len(X[0]))
b = 0

#Exercise1P1(X,y,w,b)
Exercise1P2(X,y,w,b)
#Exercise1P4(X,y,w,b)



