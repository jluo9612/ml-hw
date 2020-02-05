import xlrd
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import decimal
decimal.getcontext().prec = 4 # significant figures = 4

#parse data from source
panda_data = pd.read_csv('./pima-indians-diabetes.csv')
[(ix, k, v) for ix, row in panda_data.iterrows() for k, v in row.items()]
parsed_tuples = list(panda_data.itertuples(index = False, name = None))


def generate_targets(data): #generate targets from data
	outcomes = []
	for e in data:
		outcomes.append(e[8])
	targets = np.asarray(outcomes)
	return targets

def trim_data(data): #trim outcome column from data
	results = []
	for e in data:
		results.append(e[1:8])
	return results

def build_data(n): #build training/testing data for n samples
	training_data = []
	testing_data = []
	diabetes = []
	no_diabetes = []

	# sort data entries by outcome
	for e in parsed_tuples:
		if e[8] == 1:
			diabetes.append(e)
		elif e[8] == 0:
			no_diabetes.append(e)

	# print("Diabetes {}; no diabetes {}".format(len(diabetes), len(no_diabetes)))

	# shuffle data entries
	np.random.shuffle(diabetes)
	np.random.shuffle(no_diabetes)

	# merge data entries
	training_data.extend(diabetes[1:n]) #?
	training_data.extend(no_diabetes[1:n]) #?

	# print("n = {}, training samples = {}".format(n, len(training_data)))

	m = len(testing_data)

	# form testing data
	for e in parsed_tuples:
		if e not in training_data:
			testing_data.append(e)

	training_targets = generate_targets(training_data)
	testing_targets = generate_targets(testing_data)
	training_data_no_outcome = trim_data(training_data)
	testing_data_no_outcome = trim_data(testing_data)

	return training_targets, testing_targets, training_data_no_outcome, testing_data_no_outcome, diabetes, no_diabetes, m

def main(n):
	training_targets, testing_targets, training_data_no_outcome, testing_data_no_outcome, diabetes, no_diabetes, m = build_data(n)

	x = training_data_no_outcome
	t = training_targets.reshape(-1, 1) #1d column
	xt = np.transpose(x)
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x)), xt), t) #weight vector

	#testing
	x_test = testing_data_no_outcome
	y = np.matmul(x_test, w) # predicted results
	t_test = testing_targets.reshape(-1,1)
	#mse

	n_testcases = len(y)
	n_correct_predictions = 0
	for i, e in enumerate(y):
		if round(e[0]) == testing_targets[i]: 
			n_correct_predictions+=1

	accuracy = decimal.Decimal(n_correct_predictions) / decimal.Decimal(n_testcases)  #accuracy percentage
	# print("n {}, accuracy {}, correct pred {}, testcase {}".format(n, accuracy, n_correct_predictions, n_testcases))
	return accuracy

summation = 0
iterations = 1000
x = [40, 80, 120, 160, 200]
y = []

for n in x:
	summation = 0
	for i in range(iterations):
		accuracy = main(n) #accuracy changes every sample
		summation += accuracy

	avg = (decimal.Decimal(summation) / decimal.Decimal(iterations)) * 100
	y.append(avg)
	print("n = {}, summation = {}, avg prediction accuracy rate: {}%".format(n, summation, avg))

fig = plt.plot(x, y)
plt.xlabel('n')
plt.ylabel('Avg. Prediction Accuracy (%)')
plt.title('Avg. Prediction Accuracy vs. n')
plt.xticks(list(range(0,201,40)))
plt.show()