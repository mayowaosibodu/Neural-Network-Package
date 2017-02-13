# Works Solid.
#Adolescent Defiance - to implement

import numpy as np
import random, lib

a = np.array

class Perceptron:
	def __init__(self, mode):
		self.weights = np.random.rand(2)
		self.threshold = np.random.rand(1)[0]
		self.learning_rate = 0.08
		self.mode = mode

	def compute(self, vector,mode='classification'):
		product = np.dot(a(vector),self.weights)
		if mode == 'classification':
			if product < self.threshold: return 0 
			else: return 1
		else:
			return product

	def train(self,vectors,targets):
		for vector, target in zip(vectors, targets):
			if self.mode == 'classification':
				assigned = self.compute(vector)
				self.d = 2*(target-assigned)
			elif self.mode == 'regression':
				activation = self.compute(vector,mode='regression')
				squashed_activation = lib.sigmoid(activation, self.threshold)
				self.d = (target-squashed_activation)*lib.sigmoid_prime(activation,self.threshold)


			self.weights += self.learning_rate *self.d *vector
			self.threshold += self.learning_rate *self.d *-1

	def test(self,vectors,targets):
		results = []
		for vector in vectors:
			results.append(self.compute(vector))

		if self.mode == 'classification':
			comparisons = a(results)==a(targets)
			return len(comparisons[comparisons==True])/float(len(comparisons))
		
		elif self.mode == 'regression':
			errors = (a(targets) - a(results))**2
			return (np.mean(errors))**0.5

def run(mode, train_proportion, epochs):
	global X, Y, node
	node = Perceptron(mode)

	X, Y = lib.create_data(100, mode)
	ntrain = int(len(X)*train_proportion)
	results = []

	print 'Training...'
	for i in range(50):
		for times in range(epochs):
			node.train(X[:ntrain],Y[:ntrain])
		results.append(node.test(X[ntrain:],Y[ntrain:]))
	results = a(results)
	print 'Done. \n'

	print 'Accuracy metric:', np.mean(results)
	print 'Spread metric:', np.std(results)
	print 'Net Parameters:', node.weights, node.threshold

	lib.visualize_data(X,Y,node)

# Running model. Mode options: 'classification' or 'regression'
mode = 'regression'
# mode = 'classification'

run(mode,train_proportion = 0.5, epochs=1)
