#Good work. Very neat and modular.
#Network running in order.

# Read up on exploding and vanishing gradients.
# Regularization.


import numpy as np
import random
# from matplotlib import pyplot as plt
a = np.array

class Neuron:
	def __init__(self, input_size):
		self.weights = np.random.rand(input_size)
		self.threshold = np.random.rand(1)[0]
		self.learning_rate = 0.05
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

	def feedforward(self, vector, mode='discrete'):
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

	def fit(self, X, Y, mode='discrete', epochs=1):
		print 'Training...'
		for epoch in range(epochs):
			for i in range(len(X)):
				output = self.feedforward(X[i], mode)
				self.backprop(output, target= Y[i])
		print 'Done'

	def evaluate(self, X, Y, mode):
		outputs = []
		for x in X:
			outputs.append(self.feedforward(x)[0])

		if mode == 'discrete':
			comparisons = a(outputs)==a(Y)
			return len(comparisons[comparisons==True])/float(len(comparisons))
		
		else:
			errors = (a(Y) - a(outputs))**2
			return np.mean(errors)


me = Network(input_size= [1], layer_structure= [3,2,1])

X = np.random.rand(50)
Y = []

for x in X:
	if x <= 0.8:
		Y.append(0)
	else:
		Y.append(1)

me.fit(X,Y,epochs=100)
print 'Accuracy:', me.evaluate(X,Y,mode='discrete')