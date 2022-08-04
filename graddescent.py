import copy, math
from random import random
import numpy as np
import matplotlib.pyplot as plt

## we will be having 4 features in this model (n=4)
## we have 3 training examples (m=3)

numOfFeatures = 4 # n
numOfTrainingExamples = 3

# x_train[0] = first training example inputs
# (can be thought of as square feet, date built, etc)
x_train = np.array(
	[[2104, 5, 1, 45],
	 [1416, 3, 2, 40],
	 [852, 2, 1, 35]])

# y_train[0] = first training example output
# (can be thought of as price of house)
y_train = np.array(
	[460,
	232,
	178])

# init values for our bias and our scalar weight vectors
# b_init = 785.1811367994083
# w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

b_init = random()*50
w_init = (np.random.rand(numOfFeatures))*10
# w_init = np.array([ 0.7, 30, -100, 3])

def calcF(w, b, x_i): #calculate the output (ex price) for the params of x_i using w and b
	return np.dot(w,x_i) + b
# print(calcF(w_init, b_init, x_train[0]))

def calcJ(w,b,x,y): # calculate the cost of the entire model
	sum = 0
	for i in range(numOfTrainingExamples):
		sum += (calcF(w,b,x[i]) - y[i])**2 # calculates square cost fn
	sum *= (1/2*numOfTrainingExamples) # takes the average with respect to the # of training ex
	return sum

def calcdJ_db(w,b,x,y): # calc dJ/db (partial with respect to the basis of cost fn)
	sum = 0
	for i in range(numOfTrainingExamples):
		sum += calcF(w,b,x[i]) - y[i]
	sum *= 1/numOfTrainingExamples
	return sum

def calcdJ_dW(w,b,x,y): # calc dJ/dw, returns vector for w1 -> wn of their dJ/dwi
	# the way to get dJ/dwi can be simplified to:
	# dJ/db * wi
	dJ_db = calcdJ_db(w,b,x,y)
	dJ_dW = np.zeros(numOfFeatures) # remember W is a vector of # of features length (n)
	for i in range(numOfFeatures):
		dJ_dW[i] = dJ_db*w[i]
	return dJ_dW

# step_size = 5.0e-7
step_size = 0.000001
def calcGradDescent(w,b,x,y):
	w_new = w - step_size * calcdJ_dW(w,b,x,y)
	b_new = b - step_size * calcdJ_db(w,b,x,y)
	return (w_new, b_new)

iterations = 100000

w = w_init
b = b_init

print(f"W = {w} , b = {b}")

for i in range(iterations):
	w,b = calcGradDescent(w,b,x_train,y_train)
	if i % 10000 == 0:
		print(f"Iteration #{i}: Current Cost = {calcJ(w,b,x_train, y_train)}")

print(f"W = {w} , b = {b}")