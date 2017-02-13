
# Read up on exploding and vanishing gradients.
# Regularization.
#Sometimes works, sometimes doesnt. To understand.

# What do I do with the activations + How did I introduce nonlinearity

#Make this work for multidemensional classification and regression.
# Something is off when working in more than one dimension
# How do you draw the decision line for a multilayer net?


import numpy as np
import random, lib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.array

class Neuron:
	def __init__(self, input_size):
		self.weights = np.random.rand(input_size)
		self.threshold = np.random.rand(1)[0]
		self.learning_rate = 0.5
		self.present_vector = 0

	def compute(self, vector,mode='classification'):
		self.present_vector = a(vector)
		product = np.dot(a(vector),self.weights)
		if mode == 'classification':
			if product < self.threshold: return 0 
			else: return 1
		else:
			return product

	def backprop(self, d):
		self.d = d
		self.weights += self.learning_rate *d *self.present_vector
		self.threshold += self.learning_rate *d *-1
		return self.d

class Layer:
	def __init__(self,layer_size,input_size):
		self.neurons = []
		self.size = layer_size
		self.next_layer = 0
		for i in range(layer_size):
			neuron = Neuron(input_size)
			self.neurons.append(neuron)

	def forward_pass(self, vector, mode):
		activations = []
		for neuron in self.neurons:
			activations.append(neuron.compute(vector, mode))
		return activations

	def backprop(self, d):
		layer_backprop_ds = []
		for index in range(len(self.neurons)):
			neuron = self.neurons[index]
			nweights = []
			for n_neuron in self.next_layer.neurons:
				nweights.append(n_neuron.weights[index])
			nweights = a([nweights])
			neuron_d = sum(np.dot(nweights,d))
			layer_backprop_ds.append(neuron.backprop(neuron_d))
		return layer_backprop_ds

	def output_backprop(self, d):
		layer_backprop_ds = []
		for index in range(len(self.neurons)):
			neuron = self.neurons[index]
			layer_backprop_ds.append(neuron.backprop(d[index])) #backprop returns d[index]
		return layer_backprop_ds

class Network:
	def __init__(self,input_size, layer_structure):
		full_layer_structure = input_size+layer_structure
		self.layers = []

		for i in range(1, len(full_layer_structure)):
			layer_size = full_layer_structure[i]
			input_size = full_layer_structure[i-1]
			self.layers.append(Layer(layer_size, input_size))
		for i in range(len(self.layers)-1):
			self.layers[i].next_layer = self.layers[i+1] #do self.mode rather than passing

	def feedforward(self, vector, mode='classification'):
		to_pass = vector
		for layer in self.layers:
			to_pass= layer.forward_pass(to_pass, mode)
		return to_pass

	def backprop(self, activation, target):
		d = 2* (a(target)-a(activation))
		reversed = self.layers[:]
		reversed.reverse()			#redundant?

		for layer in reversed:
			if self.layers.index(layer) == len(self.layers)-1:
				d = a(layer.output_backprop(d))
			else:
				d = a(layer.backprop(d))

	def fit(self, X, Y, mode='classification', epochs=1):
		for epoch in range(epochs):
			for i in range(len(X)):
				output = self.feedforward(X[i], mode)
				self.backprop(output, target= Y[i])

	def evaluate(self, X, Y, mode):
		self.mode = mode
		outputs = []
		for x in X:
			outputs.append(self.feedforward(x)[0])

		if mode == 'classification':
			comparisons = a(outputs)==a(Y)
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

net = Network(input_size= [2], layer_structure= [2,1])


def run(net):

	X, Y = lib.create_data(100,'classification')

	results = []

	print 'Training...'

	for i in range(5):
		net.fit(X,Y,epochs=10)
		results.append(net.evaluate(X,Y,mode='classification'))
	results = a(results)

	print 'Done. \n'

	print 'Accuracy metric:', np.mean(results)
	print 'Spread metric:', np.std(results)
	# print 'Net Parameters:', node.weights, node.threshold

	visualize(X,Y,net)

run(net)