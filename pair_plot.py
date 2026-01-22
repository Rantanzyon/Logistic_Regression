import matplotlib.pyplot as plt
from load_csv import ft_load_csv

def main():

	try:

		data = ft_load_csv('./datasets/dataset_train.csv')
		features = data.columns[6:]
		length = len(features)
		_, ax = plt.subplots(length, length, figsize=(50,30))
		houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']


		for i in range(length):
			for j in range(length):

				ax[i, j].set_xticks([])
				ax[i, j].set_yticks([])

				for house in houses:

					if i!= j:
						feature1 = data[data['Hogwarts House'] == house][features[i]].astype(float)
						feature2 = data[data['Hogwarts House'] == house][features[j]].astype(float)
						ax[i, j].scatter(feature1, feature2, s=1)
					else:
						feature1 = data[data['Hogwarts House'] == house][features[i]].astype(float)
						ax[i, j].hist(feature1, bins=20, linewidth=1, density=True, histtype='step')


				if i == 0:
					name = features[j]
					if len(name) > 10:
						name = name[:10] + '...'
					ax[i, j].set_title(name, fontsize=8)
				if j == 0:
					name = features[i]
					if len(name) > 10:
						name = name[:10] + '...'
					ax[i, j].set_ylabel(name, fontsize=8, rotation=0, labelpad=30, va='center')
		plt.show()

	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()