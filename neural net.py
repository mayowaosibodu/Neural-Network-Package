# Works Solid.

# Next: To understand how to draw the decision line for a multilayer net.

# What happens when an activation is spot on - exactly at the threshold?
# Is it a zero, or is it a one?

# Read up on exploding and vanishing gradients.
# Regularization.

import numpy as np
import random, lib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.array


class Layer:
	def __init__(self,layer_size,input_size):
		self.weights = np.random.rand(layer_size, input_size+1)
		self.size = layer_size
		self.next_layer = 0
		self.X = 0

	def compute(self, vector):		
		weights = self.weights
		ones = [-1] if len(vector.shape)==1 else -1* np.ones((vector.shape[0],1))
		vector = np.concatenate((vector, ones), axis = len(vector.shape)-1)
		self.X = vector.copy()
		activations = np.sum(self.weights * vector, axis=1)
		print activations, np.dot(self.weights,vector)
		activations[activations>0]=1
		activations[activations<=0]=0
		return activations

	def backprop(self,d):
		delta_w = 0.01 * np.transpose(d) * self.X
		self.weights += delta_w
		weights_corresponding_to_neurons = self.weights[...,:-1]
		propagated_d = d*weights_corresponding_to_neurons
		return propagated_d

class Network:
	def __init__(self,input_size, layer_structure):
		self.layers = [Layer(layer_structure[0],input_size)]
		for size in layer_structure[1:]:
			previous_layer_size = self.layers[len(self.layers) -1].size
			self.layers.append(Layer(size, previous_layer_size))
	
	def feedforward(self, vector):
		for layer in self.layers:
			vector = layer.compute(vector)
		return vector

	def backprop(self, vector):
		self.layers.reverse()
		for layer in self.layers:
			vector = layer.backprop(vector)
		self.layers.reverse()
		return vector

	def fit(self, X, Y, mode='classification', epochs=1):
		for epoch in range(epochs):
			for i in range(len(X)):
				output = self.feedforward(X[i])
				d = 2* (Y[i]-output)
				self.backprop(d)

	def evaluate(self, X, Y, mode):
		global comparisons, outputs
		self.mode = mode
		outputs = []
		for x in X:
			outputs.append(self.feedforward(x))

		if mode == 'classification':
			comparisons = a(outputs).flatten()==a(Y)
			return len(comparisons[comparisons==True])/float(len(comparisons))
		
		else:
			errors = (a(Y) - a(outputs))**2
			return np.mean(errors)

def visualize(X,Y,node):
	fig = plt.figure()
	graph = fig.add_subplot(111,projection='3d')

	if node.mode == 'classification':
		graph.scatter(a(X)[...,0],a(X)[...,1],c=Y)
		# predicted_Y = (-node.weights[0]/node.weights[1])*np.linspace(-5,15,20) + node.threshold/node.weights[1]
		# graph.plot(np.linspace(-5,15,20), predicted_Y)
	
	# elif node.mode == 'regression':
	# 	predicted_Y = node.weights[0]*x + node.weights[1]*y - node.threshold
	# 	predicted_Y -= np.mean(predicted_Y)
	# 	predicted_Y /= np.std(predicted_Y)
	# 	graph.scatter(X[...,0],X[...,1],predicted_Y, c='g')
	# 	graph.scatter(X[...,0],X[...,1],Y, c='b') #Hm surface plot doesnt seem to work
	plt.show()

net = Network(input_size= 2, layer_structure= [2,1])

def run(net):

	X, Y = lib.create_data(50,2, 'classification')

	results = []

	print '\n \n Training... \n '

	for i in range(1):
		net.fit(X,Y,epochs=10)
		results.append(net.evaluate(X,Y,mode='classification'))
	results = a(results)

	print 'Done. \n'

	print 'Accuracy metric:', np.mean(results)
	print 'Spread metric:', np.std(results)

	visualize(X,Y,net)

run(net)