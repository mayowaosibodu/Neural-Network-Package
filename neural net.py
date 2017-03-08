# Regression curve is being fit.
# Work on fitting perfectly.

import numpy as np
import random, lib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.array


class Layer:
	def __init__(self,layer_size,input_size,mode):
		self.weights = np.random.rand(layer_size, input_size+1)
		self.mode = mode
		self.size = layer_size
		self.next_layer = 0
		self.X = 0

	def compute(self, vector):
		global activations		
		weights = self.weights
		ones = [-1] if len(vector.shape)==1 else -1* np.ones((vector.shape[0],1))
		vector = np.concatenate((vector, ones), axis = len(vector.shape)-1)
		self.X = vector.copy()
		activations = self.weights * vector
		if self.mode == 'classification':
			activations = np.sum(activations, axis=1)
			activations[activations>0]=1
			activations[activations<=0]=0

		if self.size != 1:  #To check if layer is output
			weight_activations = np.sum(activations[...,:-1], axis=1)
			thresholds = activations[...,-1]
			squashed_activations = []

			for output,threshold in zip(weight_activations,thresholds):
				squashed_activations.append(lib.sigmoid(output, 0))#threshold))
			activations = a(squashed_activations)
		else:
			activations = np.sum(activations, axis=1)

		return activations


	def backprop(self,d):

		delta_w = 0.0001 * np.transpose(d) * self.X
		self.weights += delta_w
		weights_corresponding_to_neurons = self.weights[...,:-1]
		propagated_d = np.sum(d*np.transpose(weights_corresponding_to_neurons))
		# print 'Propagated d', propagated_d #Hm I just noticed that the net only learns for very few x-y data pairs. Hm why is this?
										   #And how could it be taken advantage of? Mm, how?

		return propagated_d

class Network:
	def __init__(self,input_size, layer_structure, mode):
		self.layers = [Layer(layer_structure[0],input_size, mode)]
		self.mode = mode
		for size in layer_structure[1:]:
			previous_layer_size = self.layers[len(self.layers) -1].size
			self.layers.append(Layer(size, previous_layer_size, mode))
	
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

	def fit(self, X, Y, epochs=1):
		for epoch in range(epochs):
			print '\n'
			print 'Epoch:', epoch
			print 'Accuracy', self.evaluate(X,Y)
			for i in range(len(X)):
				output = self.feedforward(X[i])
				# print X[i], Y[i], output
				d = 2* (Y[i]-output)* lib.sigmoid_prime(output,threshold=0)
				self.backprop(d)

	def evaluate(self, X, Y):
		global comparisons, outputs
		outputs = []
		for x in X:
			outputs.append(self.feedforward(x))

		if self.mode == 'classification':
			comparisons = a(outputs).flatten()==a(Y)
			return len(comparisons[comparisons==True])/float(len(comparisons))
		
		else:
			errors = (a(Y) - a(outputs))**2
			return np.mean(errors)
net = Network(input_size= 2, layer_structure= [10,1], mode='regression')

def run(net, input_size, epochs):
	global X, Y

	X, Y = lib.create_data(200,input_size,net.mode)

	results = []

	print '\n \n Training... \n '

	for i in range(1):
		net.fit(X,Y,epochs)
		results.append(net.evaluate(X,Y))
	results = a(results)

	print 'Done. \n'

	print 'Accuracy metric:', np.mean(results)
	print 'Spread metric:', np.std(results)

	# visualize(X,Y,net)
	lib.visualize_net(X,Y,net)

run(net, input_size=2, epochs=100)