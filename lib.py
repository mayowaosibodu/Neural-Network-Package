
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.array

def sigmoid(input, threshold):
		output = 1/(1+ np.e**-(input-threshold))
		return output

def sigmoid_prime(input, threshold):
		output = sigmoid(input,threshold)*(1-sigmoid(input,threshold))
		return output

def create_data(size, mode):
	# Create X and Y
	X = np.random.rand(size,2) 

	if mode == 'classification':
		Y = [0 if 2*x[0] + 5*x[1] <= 3.5 else 1 for x in X]
	elif mode == 'regression':
		Y = a([2*x[0] + 5*x[1] - 3.5 for x in X]) + np.random.rand(size)*2

	# normalize
	X -= np.mean(X)
	X /= np.std(X)

	if mode== 'regression':
		Y -= np.mean(Y)
		Y /= np.std(Y)

	return X, Y

def visualize_data(X,Y,node):
	fig = plt.figure()
	graph = fig.add_subplot(111,projection='3d')

	if node.mode == 'classification':
		graph.scatter(a(X)[...,0],a(X)[...,1],c=Y)
		predicted_Y = (-node.weights[0]/node.weights[1])*np.linspace(-5,15,20) + node.threshold/node.weights[1]
		graph.plot(np.linspace(-5,15,20), predicted_Y)
	
	elif node.mode == 'regression':
		predicted_Y = node.weights[0]*X[...,0] + node.weights[1]*X[...,1] - node.threshold
		predicted_Y -= np.mean(predicted_Y)
		predicted_Y /= np.std(predicted_Y)
		graph.scatter(X[...,0],X[...,1],predicted_Y, c='g')
		graph.scatter(X[...,0],X[...,1],Y, c='b') #Hm surface plot doesnt seem to work
	plt.show()

def visualize_error_function(X,Y,node):
	fig = plt.figure()
	graph = fig.add_subplot(111,projection='3d')
	graph.scatter(X[...,0],X[...,1],Y, c='g')
	plt.show()