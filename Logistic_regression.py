import pandas
import numpy
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class Logistic_Regression():

    def __init__(self, dataset: pandas.DataFrame):

        self._dataset = dataset

    def get_stats(self) -> dict:

        stats = {}

        for column in self.X_train.iloc(axis=1):
            mean = sum(column) / len(column)
            var = sum([(x - mean) ** 2 for x in column]) / (len(column) - 1)
            std = var ** 0.5
            stats[column.name] = { 'mean': mean, 'std': std }

        return stats
    
    def normalize(self, train):

        X = numpy.array(train)
        mean = numpy.array([feature['mean'] for feature in self._stats.values()])
        std = numpy.array([feature['std'] for feature in self._stats.values()])
        
        return (X - mean) / std


    def setup(self, label: str, features: list, frac=0.8):

        self._label = label # Hogwarts House
        self._labels = list(set(self._dataset[label])) # <list> ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
        self._features = features # <list> [Arithmancy   Astronomy  ...  Muggle Studies  Ancient Runes  History of Magic  Transfiguration    Potions  Care of Magical Creatures     Charms  Flying]

        self._dataset = self._dataset.dropna()

        # self._train = self._dataset.iloc[:int(len(self._dataset) * 80 / 100)]
        # self._test = self._dataset.iloc[int(len(self._dataset) * 80 / 100):]
        self._train = self._dataset.sample(frac=0.8)
        self._test = self._dataset.drop(self._train.index)

        self.Y_train = self._train[label]
        self.Y_test = self._test[label]

        self.X_train = self._train[features]
        self.X_test = self._test[features]

        self._stats = self.get_stats()

        self.Xn_train = self.normalize(self.X_train)
        self.Xn_test = self.normalize(self.X_test)
    

    def ft_init(self, init: str, ) -> dict:

        height = len(self._labels)
        width = self.Xn_train.shape[1]
        params = {}


        match init:

            case 'zeros':
                params['W'] = numpy.zeros((height, width))
                params['b'] = numpy.zeros((height, 1))
            case _:
                params['W'] = numpy.random.randn(height, width)
                params['b'] = numpy.random.randn(height, 1)
            
        return params

    def ft_activation(self, X, params) -> dict:

        activations = {}

        Z = params['W'].dot(X.T) + params['b']
        activations['A'] = 1 / (1 + numpy.exp(-Z))

        return activations
    
    def ft_gradients(self, X, Y, activations) -> dict:

        m = X.shape[0]
        gradients = {}

        gradients['W'] = (1 / m) * numpy.dot(activations['A'] - Y, X)
        gradients['b'] = (1 / m) * numpy.sum(activations['A'] - Y, axis=1, keepdims=True)

        return gradients
    
    def ft_update(self, gradients, params, lr) -> dict:

        params['W'] -= lr * gradients['W']
        params['b'] -= lr * gradients['b']

        return params
    
    def ft_testing(self, X, Y, params):

        activations = self.ft_activation(X, params)

        count = 0
        for i, stud in enumerate(activations['A'].T):

            prediction = self._labels[numpy.argmax(stud)]
            reality = Y.iloc[i]

            if prediction == reality:
                count += 1

        accuracy = count * 100 / len(X)
        print(f'Accuracy : {accuracy:.2f}%')


    def train(self, optimization='GD', init='random', epochs=5000, learning_rate=0.1):

        X = self.Xn_train
        Y = numpy.array([[1 if x_label == label else 0 for x_label in self.Y_train] for label in self._labels])

        params = self.ft_init(init)

        for epoch in tqdm(range(epochs)):

            activations = self.ft_activation(X, params)
            gradients = self.ft_gradients(X, Y, activations)
            params = self.ft_update(gradients, params, learning_rate)


        # ============== TESTING

        self.ft_testing(self.Xn_test, self.Y_test, params)


def main():

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--file', default='./datasets/dataset_train.csv')
        parser.add_argument('--conf', default='conf.json')
        args = parser.parse_args()

        dataset = pandas.read_csv(args.file)
        with open(args.conf, 'r') as file:
            conf = json.load(file)

        logreg = Logistic_Regression(dataset, conf)

        features = dataset.columns[6:]
        features_filtered = features.drop(['Arithmancy', 'Care of Magical Creatures'])

        logreg.setup('Hogwarts House', features_filtered)
        logreg.train(init='random', optimization='GD', epochs=10000, learning_rate=0.1)
        # init: random, zeros
        # optimization : GD, SGD, BGD, MBGD

    except BaseException as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
