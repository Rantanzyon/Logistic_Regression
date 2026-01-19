from load_csv import ft_load_csv
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.linear_model import LogisticRegression


def main():

	try:

		data = ft_load_csv('./datasets/dataset_test.csv')
		result = ft_load_csv('result.csv')
		features = data.columns[6:]
		houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']

		mean_row = result[result['Hogwarts House'] == 'mean'].iloc[0]
		std_row = result[result['Hogwarts House'] == 'std'].iloc[0]

		# remplacer nan par la mean du training
		data[features] = data[features].fillna(mean_row[features])
		# print(data[features])

		# normaliser
		data[features] = (data[features] - mean_row[features]) / std_row[features]
		# print(data[features])

		X = np.array(data[features])
		# print(X)

		predictions = []

		for i in range(len(X)):

			probas = []

			for house in houses:

				row = result[result['Hogwarts House'] == house].iloc[0]

				W = row[features].to_numpy()
				b = row['b']


				Z = X[i].dot(W) + b
				p = 1 / (1 + np.exp(-Z))

				probas.append(p)

			predictions.append(houses[np.argmax(probas)])

		# print(predictions)

		# ====================================

		





		









		# for house in houses:












		

	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()

