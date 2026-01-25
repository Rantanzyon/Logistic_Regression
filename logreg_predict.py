import numpy as np
import json
import pandas as pd

def ft_mean(training: pd.DataFrame, features: list) -> dict:

	mean = {}
	for feature in features:
		mean[feature] = training['Gryffindor'][feature]['mean']
	return mean

def ft_std(json_data: pd.DataFrame, features: list) -> dict:

	std = {}
	for feature in features:
		std[feature] = json_data['Gryffindor'][feature]['std']
	return std

def ft_normalize(dataset: pd.DataFrame, mean: dict, std: dict) -> np.array:

	features = dataset.columns
	mean = np.array(list(mean.values()))
	std = np.array(list(std.values()))
	dataset = np.array(dataset[features])
	dataset_norm = (dataset - mean) / std
	return dataset_norm

def get_parameters(training: pd.DataFrame, features: list):

	houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
	Ws = []
	Bs = []
	for house in houses:

		W = []
		for feature in features:
			W.append(training[house][feature]['w'])
			b = training[house][feature]['b']

		# W = [training[house][feature]['w'] for feature in features]
		Ws.append(W)
		Bs.append(b)
	return (np.array(Ws), np.array(Bs))

def ft_biais(training: pd.DataFrame, features: list):

	houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
	Bs = []
	for house in houses:
		Bs.append(training[house]['b'])
	return np.array(Bs)


def main():

	try:

		with open('datajson.json', 'r') as file:
			training = json.load(file)


		houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
		dataset = pd.read_csv('./datasets/dataset_test.csv')
		dataset = dataset.drop(columns=['Arithmancy', 'Care of Magical Creatures'])
		features = dataset.columns[6:]
		dataset = dataset[features]

		mean = ft_mean(training, features)
		std = ft_std(training, features)

		# fill nan with mean
		dataset = dataset.fillna(value=mean)

		# normalize (x - mean) / std
		X = ft_normalize(dataset, mean, std)

		W, b = get_parameters(training, features)
		# W = ft_weights(training, features)
		# b = ft_biais(training, features)

		Z = X.dot(W.T) + b
		A = 1 / (1 + np.exp(-Z))

		with open('houses.csv', 'w') as file:

			file.write('Index,Hogwarts House\n')
			for index, probs in enumerate(A):

				imax = np.argmax(probs)
				file.write(f'{index},{houses[imax]}\n')

		
	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()

