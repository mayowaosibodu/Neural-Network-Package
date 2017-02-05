
#Sweet sweet.
#There seems to be a constant numerical difference between the data surface
#and the fitted surface. Check out the reason for this.


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
			return np.mean(errors)

def show(X,Y,me):
	if me.mode == 'classification':
		vis = plt.figure()
		graph = vis.add_subplot(111)
		graph.scatter(a(X)[...,0],a(X)[...,1],c=Y)
		graph.plot(np.linspace(-5,15,20),(-me.weights[0]/me.weights[1])*np.linspace(-5,15,20) + me.threshold/me.weights[1])
	
	elif me.mode == 'regression':
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		x = X[...,0]#np.linspace(0,10,num=100)
		y = X[...,1]#np.random.rand(100)*10
		z = me.weights[0]*x + me.weights[1]*y - me.threshold/2
		ax.scatter(x,y,z, c='g')
		ax.scatter(X[...,0],X[...,1],Y, c='b') #Hm surface plot doesnt seem to work
		plt.show()

def evaluate(mode):
	global X,Y,graph, node

	node = Perceptron(mode)
	X = np.random.rand(100,2)*10
	Y=[]

	if mode == 'classification':
		for i in X:
			if 2*i[0] + 5*i[1] <= 35:
				Y.append(0)
			else:
				Y.append(1)

	elif mode == 'regression':
		for i in X:
			Y.append(2*i[0] + 5*i[1] - 35)

	X = a(X)
	Y = a(Y)

	mean = np.mean(X)
	std = np.std(X)

	X -= mean
	X /= std

	Y -= mean
	Y /= std

	for j in range(50):
		node.train(X,Y)

	return node.test(X,Y)


# Running model. Mode options: 'classification' or 'regression'

mode = 'regression'
results = []

for i in range(100):
	results.append(evaluate(mode))
results = a(results)

print 'Accuracy Metric:', np.mean(results)
print 'Spread Metric:', np.std(results)
print 'Net Parameters:', node.weights, node.threshold

show(X,Y,node)