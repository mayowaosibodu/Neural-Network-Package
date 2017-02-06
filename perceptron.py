
#Beautiful.

import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array

class Perceptron:
	def __init__(self, mode):
		self.weights = np.random.rand(2)
		self.threshold = np.random.rand(1)[0]
		self.learning_rate = 0.08
		self.mode = mode

	def sigmoid(self, input, threshold):
		denominator = 1+ np.e**-(input-threshold)
		output = 1/denominator
		return output

	def sigmoid_prime(self, input, threshold):
		output = self.sigmoid(input,threshold)*(1-self.sigmoid(input,threshold))
		return output

	def compute(self, vector,mode='discrete'):
		product = np.dot(a(vector),self.weights)
		if mode == 'discrete':
			if product < self.threshold:
				return 0
			else:
				return 1
		else:
			return product

	def train(self,vectors,targets):
		for i in range(len(vectors)):
			vector = a(vectors[i])
			target = a(targets[i])
			if self.mode == 'classification':
				assigned = self.compute(vector)
				self.d = 2*(target-assigned)
			elif self.mode == 'regression':
				activation = self.compute(vector,mode='continuous')
				squashed_activation = self.sigmoid(activation, self.threshold)
				self.d = (target-squashed_activation)*self.sigmoid_prime(activation,self.threshold)


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

def show(X,Y,node):
	if node.mode == 'classification':
		vis = plt.figure()
		graph = vis.add_subplot(111)
		graph.scatter(a(X)[...,0],a(X)[...,1],c=Y)
		graph.plot(np.linspace(-5,15,20),(-node.weights[0]/node.weights[1])*np.linspace(-5,15,20) + node.threshold/node.weights[1])
	
	elif node.mode == 'regression':
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		x = X[...,0]
		y = X[...,1]
		z = node.weights[0]*x + node.weights[1]*y - node.threshold
		z -= np.mean(z)
		z /= np.std(z)
		ax.scatter(x,y,z, c='g')
		ax.scatter(X[...,0],X[...,1],Y, c='b') #Hm surface plot doesnt seem to work
	plt.show()

def evaluate(mode):
	global node, X, Y
	node = Perceptron(mode)
	X = np.random.rand(100,2)
	Y=[]

	if mode == 'classification':
		for i in X:
			# Function to be fitted by model
			if 2*i[0] + 5*i[1] <= 3.5:
				Y.append(0)
			else:
				Y.append(1)

	elif mode == 'regression':
		for i in X:
			# Function to be fitted by model
			Y.append(2*i[0] + 5*i[1] - 3.5)

	X = a(X)
	Y = a(Y)

	X -= np.mean(X)
	X /= np.std(X)

	if mode== 'regression':
		Y -= np.mean(Y)
		Y /= np.std(Y)

	for j in range(2):
		node.train(X,Y)

	return node.test(X,Y)


# Running model. Mode options: 'classification' or 'regression'

mode = 'regression'
# mode = 'classification'
results = []

print 'Training...'

for i in range(50):
	results.append(evaluate(mode))
results = a(results)

print 'Done. \n'

print 'Accuracy metric:', np.mean(results)
print 'Spread metric:', np.std(results)
print 'Net Parameters:', node.weights, node.threshold

show(X,Y,node)