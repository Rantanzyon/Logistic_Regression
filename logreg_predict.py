from load_csv import ft_load_csv
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import pandas as pd

def main():

	try:

		with open('datajson.json', 'r') as file:
			training = json.load(file)



		# print(training)

		houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
		dataset = pd.read_csv('./datasets/dataset_test.csv')

		

		dataset = dataset.drop(columns=['Arithmancy', 'Care of Magical Creatures'])







		
	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()

