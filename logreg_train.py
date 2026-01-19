from load_csv import ft_load_csv
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score



def main():

	try:

		data = ft_load_csv('./datasets/dataset_train.csv')

		features = data.columns[6:]
		houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']
		data[features] = data[features].fillna(data[features].mean())

		dataset_size = len(data)
		print(dataset_size)
		

		set_train = data.iloc[:int(len(data) * 80 / 100)]
		set_test = data.iloc[int(len(data) * 80 / 100):]


		# normaliser
		mean_train = set_train[features].mean()
		std_train = set_train[features].std()
		print('a')

		# print(mean_train)
		# print(std_train)

		# features matrice
		X = np.array((set_train[features] - mean_train) / std_train)
		# print(f'X: {X.shape}')


		# vector weights
		W = np.array([0] * 13)
		# print(f'W: {W.shape}')


		# biais
		b = 0
		# print(b)

		# model Z
		Z = X.dot(W) + b
		# print(Z)
		# print(f'Z: {Z.shape}')


		# model A: sigmoid
		A = 1 / (1 + np.exp(-Z))
		# print(f'A: {A.shape}')



		m = len(X)
		# print(m)

		epochs = 1000
		lr = 0.05

		result = {}
		W_all = []
		b_all = []
		for house in houses:
			Y = np.array([1 if set_train.loc[index, 'Hogwarts House'] == house else 0 for index in set_train.index])
			result[house] = {}
			for epoch in range(epochs):

				Z = X.dot(W) + b
				A = 1 / (1 + np.exp(-Z))
				grad_W = (1 / m) * (np.transpose(X)).dot((A - Y))
				grad_b = (1 / m) * np.sum(A - Y)

				W = W - lr * grad_W
				b = b - lr * grad_b
			
			result[house]['W'] = W
			result[house]['b'] = b
			W_all.append(W)
			b_all.append(b)

		print('================')
		W_all = np.array(W_all)
		b_all = np.array(b_all)


		print('==============================')
		print(W_all)
		print(b_all)

		# ================================
		# TESTING ACCURACY

		print('a')

		X_test = np.array((set_test[features] - mean_train) / std_train)
		print(X_test.shape)

		Z = X_test.dot(W_all.T) + b_all
		A = 1 / (1 + np.exp(-Z))

		pred = np.array(houses)[np.argmax(A, axis=1)]
		print(pred)

		print('===============================')

		real_test = set_test['Hogwarts House']
		print(real_test)

		print('======')

		print(accuracy_score(real_test, pred))




		# # ================================
		# # MY PREDICTION

		# df_test = ft_load_csv('./datasets/dataset_test.csv')

		# # Remplacer les nan par la mean
		# df_test[features] = df_test[features].fillna(mean_train[features])

		# # normaliser
		# df_test[features] = (df_test[features] - mean_train[features]) / std_train[features]
		# print(df_test[features].shape)


		# X_test = df_test[features].to_numpy()

		# Z = X_test.dot(W_all.T) + b_all
		# A = 1 / (1 + np.exp(-Z))

		# pred = np.array(houses)[np.argmax(A, axis=1)]
		# # print(pred)

		# # ================================
		# # SKLEARN PRED

		clf = LogisticRegression(
			max_iter=1000
		)

		clf.fit(X, data['Hogwarts House'])

		sk_pred = clf.predict(X_test)
		# print(sk_pred)
		print(accuracy_score(real_test, sk_pred))

		# same = pred == sk_pred

		# print("Même prédiction :", np.sum(same), "/", len(same))
		# print("Différences     :", np.sum(~same))





			

		# # print(W)
		# # print(b)

		# Z = X.dot(W) + b
		# A = 1 / (1 + np.exp(-Z))

		# pred = (A >= 0.5).astype(int)
		# accuracy = np.mean(pred == Y)

		# # print("accuracy Gryffindor:", accuracy)

		# x = np.array([60888.0,-482.1390963522175,-7.327170322866657,4.821390963522175,-6.6110000000000015,-525.2725025237612,359.14817180301407,5.197858179249288,1070.7121427170061,10.032461125685584,-2.1852199688256664,-253.62839,-126.66])

		# x = np.where(np.isnan(x), mean_train, x)

		# x = (x - mean_train) / std_train


		# probs = []
		# for house in houses:

		# 	z = x.dot(result[house]['W']) + result[house]['b']
		# 	probs.append(1 / (1 + np.exp(-z)))

		# print(houses[np.argmax(probs)])

		# # ======================================

		# y = data['Hogwarts House']
		# clf = LogisticRegression(
		# 	max_iter=1000
		# )

		# clf.fit(X, y)


		# mean_train = mean_train.to_numpy()
		# std_train = std_train.to_numpy()

		# # ton vecteur élève
		# x_test = np.array([60888.0,-482.1390963522175,-7.327170322866657,4.821390963522175,-6.6110000000000015,-525.2725025237612,359.14817180301407,5.197858179249288,1070.7121427170061,10.032461125685584,-2.1852199688256664,-253.62839,-126.66])

		# # remplir NaN
		# x_test = np.where(np.isnan(x_test), mean_train, x_test)

		# # normalisation
		# x_test = (x_test - mean_train) / std_train

		# # reshape pour sklearn
		# x_test = x_test.reshape(1, -1)

		# # prédiction sklearn
		# sk_pred = clf.predict(x_test)

		# print("sklearn prediction:", sk_pred)





		# print(result)

		with open('result.csv', 'w') as file:

			file.write(f'Hogwarts House,b,{','.join(data[features])}\n')

			for house in houses:

				file.write(f'{house},{result[house]['b']},{','.join(result[house]['W'].astype(str))}\n')

			file.write(f'mean,,{','.join(mean_train.astype(str))}\n')
			file.write(f'std,,{','.join(std_train.astype(str))}\n')
			# index = 0
			# for feature in features:

			# 	for house in houses:

			# 		w, b = result[feature][house]
			# 		file.write(f'{index},{feature},{house},{w},{b}\n')
			# 		index += 1
			# n = 0

			# print(f'Gryffindor: {1 / (1 + np.exp(-(wg*n+bg)))}')
			# print(f'Hufflepuff: {1 / (1 + np.exp(-(wh*n+bh)))}')
			# print(f'Slytherin: {1 / (1 + np.exp(-(ws*n+bs)))}')
			# print(f'Ravenclaw: {1 / (1 + np.exp(-(wr*n+br)))}')






		

	except BaseException as error:
		print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
	main()

