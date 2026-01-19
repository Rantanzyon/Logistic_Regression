import matplotlib.pyplot as plt
from load_csv import ft_load_csv

def main():

	try:

		data = ft_load_csv('./datasets/dataset_train.csv')
		features = data.columns[6:]
		houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']

		for i, feature in enumerate(features):
			print(f'{i}: {feature}')

		print()
		feature1 = features[int(input('Enter an index of the x-axis feature: '))]
		feature2 = features[int(input('Enter an index of the y-axis feature: '))]

		for house in houses:

			feature1_data = data[data['Hogwarts House'] == house][feature1].astype(float)
			feature2_data = data[data['Hogwarts House'] == house][feature2].astype(float)
			plt.scatter(feature1_data, feature2_data, label=house)

		plt.xlabel(feature1)
		plt.ylabel(feature2)
		plt.legend()
		plt.show()

	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()