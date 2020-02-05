# Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. Each
# data sample has four attributes: sepal length, sepal width, petal length, and petal width.

# Implement the K-means clustering algorithm to group the samples into K=3 clusters. Randomly choose three
# samples as the initial cluster centers. Calculate the objective function value J as defined in Problem 1 after the
# assignment step in each iteration. Exit the iterations if the following criterion is met and Iter is the iteration number. Plot the objective function value J versus the iteration number
# Iter. Comment on the result. Attach the code at the end of the homework.

import sys
import xlrd
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def parse_data(path):
	# xlsx = xlrd.open_workbook(path)
	raw = xlrd.open_workbook(path).sheet_by_index(0)
	data, labels = [], []
	for row_i in range(raw.nrows-1):
		row = raw.row_values(row_i+1)
		data.append(row[1:-1])
		labels.append(row[-1])

	return data, labels

EPSILON = 1e-5

class KMeans:
	def __init__(self, data, labels):
		self.data = data
		self.cluster_labels = list(set(labels)) # k=3 types 
		self.prev_cost = float('inf') # initial cost = infinity
		self.means = random.sample(self.data, len(self.cluster_labels))

	def start(self):
		data = np.array(self.data)
		clusters = np.zeros(len(data))
		means = np.array(self.means)
		costs = []

		while True:
			cost = 0
			#Assign data point to closest cluster
			for i in range(len(data)):
				distances = np.linalg.norm(data[i]-means, axis=1)
				cluster = np.argmin(distances)
				clusters[i] = cluster
				cost += distances[cluster]
			
			costs.append(cost)
			prev_means = np.copy(means) #deep copy matrix
			
			#Recalculate cluster center for each cluster
			for c in self.cluster_labels:
				points = [data[i] for i in range(len(data)) if clusters[i] == c]
				means[int(c)] = np.mean(points, axis=0)

			delta = np.linalg.norm(prev_means - means)
			if delta < EPSILON:
				break

		return costs

def get_costs(data, labels):
	k_means = KMeans(data, labels)
	return k_means.start()

def plot(costs):
	plt.plot(range(len(costs)), costs, 'r-o') # number of iterations
	plt.axis([0, len(costs), 0, max(costs) * 1.5]) # ticks and label values
	plt.xlabel('Iteration')
	plt.ylabel('J (Cost)')
	plt.show()

if __name__ == '__main__':
	path = sys.argv[1]

	#Parse data
	data, labels = parse_data(path)

	#Get costs while kmeans runs
	costs = get_costs(data, labels)
	print("costs ", costs)

	#Plot J values vs. iterations
	plot(costs)