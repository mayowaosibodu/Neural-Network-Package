# Past Lib

import numpy as np
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.array

def sigmoid(input, threshold):
		output = 1/(1+ np.e**-(input-threshold))
		return output

def sigmoid_prime(input, threshold):
		output = sigmoid(input,threshold)*(1-sigmoid(input,threshold))
		return output

def create_data(size, dimension, mode):
	
	X = np.random.rand(size,dimension)*2
	coefficients = [2,1]#5,9,6]#Should make generation automatic later. np.random.rand(size,dimension)
	Y = []

	for point in X:
		y = 0
		for xvalue in point:
			y += xvalue**coefficients[list(point).index(xvalue)]

		if mode == 'classification':
			y = [0 if y <= 2.25 else 1]
		elif mode == 'regression':
			y = a([y])# + np.random.rand(size)*2
		# print point, y
		# print '\n \n'
		Y.append(y)

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

def visualize_net(X,Y,net):
	fig = plt.figure()
	graph = fig.add_subplot(111,projection='3d')

	if net.mode == 'classification':
		graph.scatter(a(X)[...,0], a(X)[...,1], c=Y)
		for layer in net.layers:
			for node_weights in layer.weights:
				predicted_Y = (-node_weights[0]/node_weights[1])*np.linspace(-5,15,20) + node_weights[2]/node_weights[1]
				graph.plot(np.linspace(-5,15,20), predicted_Y)

	else:
		predicted_Y = []
		for x in X:
			predicted_Y.append(net.feedforward(x))
		# Generalize plot to work with arbitrary dimensions.
		# graph.plot_surface(X[...,0],X[...,1],predicted_Y, cmap= cm.coolwarm)
		# graph.plot_surface(X[...,0],X[...,1],Y, cmap= cm.coolwarm)

		graph.scatter(X[...,0],X[...,1],predicted_Y, c='g')
		graph.scatter(X[...,0],X[...,1],Y, c='b')
		
		# graph.scatter(X[...,0],predicted_Y, c='g')
			# X[...,1],predicted_Y, c='g')
		# graph.scatter(X[...,0],Y, c='b')
			# X[...,1],Y, c='b')

	plt.show()