#Good work. Very neat and modular.
#Forward pass done.
#Working on backprop.
#Some issue with array multiplication in backprop(). To fix.      


import numpy as np
import random
from matplotlib import pyplot as plt
a = np.array

class Perceptron:
	def __init__(self, input_size):
		self.weights = np.random.rand(input_size)
		self.threshold = np.random.rand(1)[0]
		self.learning_rate = 0.8
		self.present_vector = 0

	def sigmoid(self, input, threshold):
		denominator = 1+ np.e**-(input-threshold)
		output = 1/denominator
		return output
		
	def sigmoid_prime(self, input, threshold):
		output = self.sigmoid(input,threshold)*(1-self.sigmoid(input,threshold))
		return output

	def compute(self, vector,mode='discrete'):
		self.present_vector = a(vector)
		product = np.dot(a(vector),self.weights)
		if mode == 'discrete':
			if product < self.threshold:
				return 0
			else:
				return 1
		else:
			return product

	def backprop(self, d):
		self.backprop_d = self.learning_rate *d *self.present_vector
		self.weights += self.backprop_d
		self.threshold += self.learning_rate *d *-1
		return self.backprop_d

	def test(self,vectors,targets):
		results = []
		for vector in vectors:
			results.append(self.compute(vector))
		comparisons = a(results)==a(targets)
		return len(comparisons[comparisons==True])/float(len(comparisons))

class Layer:
	def __init__(self,layer_size,input_size):
		self.perceptrons = []
		self.size = layer_size
		self.next_layer = 0
		for i in range(layer_size):
			perceptron = Perceptron(input_size)
			self.perceptrons.append(perceptron)

	def forward_pass(self, vector):
		activations = []
		for perceptron in self.perceptrons:
			activations.append(perceptron.compute(vector))
		return activations

	def backprop(self, d):
		layer_backprop_ds = []
		for index in range(len(self.perceptrons)):
			perceptron = self.perceptrons[index]
			nweights = []
			for n_perceptron in self.next_layer.perceptrons:
				nweights.append(n_perceptron.weights[index])
			nweights = a([nweights])#.T
			print nweights.shape, d.shape
			perceptron_d = np.sum(nweights* d, axis=0)
			layer_backprop_ds.append(perceptron.backprop(perceptron_d))
			# some issue with array multiplication here
		return layer_backprop_ds

	def output_backprop(self, d):
		layer_backprop_ds = []
		for index in range(len(self.perceptrons)):
			perceptron = self.perceptrons[index]
			layer_backprop_ds.append(perceptron.backprop(d[index]))
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
			self.layers[i].next_layer = self.layers[i+1]

	def feedforward(self, vector):
		to_pass = vector
		for layer in self.layers:
			to_pass= layer.forward_pass(to_pass)
			print self.layers.index(layer), to_pass
		return to_pass
			# pass #haha. (forward pass).

	def backprop(self, activation, target):
		global d
		d = 2* (a(target)-a(activation))
		reversed = self.layers[:]
		reversed.reverse()			#redundant?

		for layer in reversed:
			print 'Backpropping for layer', self.layers.index(layer)
			if self.layers.index(layer) == len(self.layers)-1:
				d = a(layer.output_backprop(d))
			else:
				d = a(layer.backprop(d))

me = Network(input_size= [5], layer_structure= [3,4,2])