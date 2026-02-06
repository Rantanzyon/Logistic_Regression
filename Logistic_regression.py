import pandas
import numpy
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random

class Logistic_Regression():

    def __init__(self, conf_file):

        with open(conf_file, 'r') as file:
            conf = json.load(file)

        # DATA_FILE
        if not conf.get('data_file'):
            raise ValueError(f'{conf_file}: data_file: key is missing.')
        else:
            self._dataset = pandas.read_csv(conf['data_file'])

        # ACTIVATION
        if not conf.get('activation'):
            self.activation = 'sigmoid'
        else:
            expected_values = ['sigmoid']
            if conf['activation'] not in expected_values:
                raise ValueError(f'{conf_file}: activation: expected {expected_values}.')
            self.activation = conf['activation']

        # WEIGHTS_INIT
        if not conf.get('weights_init'):
            self.weights_init = 'zeros'
        else:
            expected_values = ['zeros', 'random']
            if conf['weights_init'] not in expected_values:
                raise ValueError(f'{conf_file}: weights_init: expected {expected_values}.')
            self.weights_init = conf['weights_init']

        # LOSS
        if not conf.get('loss'):
            self.loss = 'BCE'
        else:
            expected_values = ['BCE']
            if conf['loss'] not in expected_values:
                raise ValueError(f'{conf_file}: loss: expected {expected_values}.')
            self.loss = conf['loss']

        # EPOCHS
        if not conf.get('epochs'):
            self.epochs = 10000
        else:
            if not isinstance(conf['epochs'], int) or conf['epochs'] < 1:
                raise ValueError(f'{conf_file}: epochs: expected a positive integer.')
            self.epochs = conf['epochs']

        # LEARNING_RATE
        if not conf.get('learning_rate'):
            self.learning_rate = 0.1
        else:
            if not isinstance(conf['learning_rate'], float) or conf['learning_rate'] < 0:
                raise ValueError(f'{conf_file}: learning_rate: expected a positive float.')
            self.learning_rate = conf['learning_rate']

        # OPTIMIZATION
        if not conf.get('optimization'):
            self.optimization = 'GD'
        else:
            expected_values = ['GD', 'SGD', 'BGD', 'MBGD']
            if conf['optimization'] not in expected_values:
                raise ValueError(f'{conf_file}: optimization: expected {expected_values}.')
            self.optimization = conf['optimization']

        # BATCH_SIZE
        if not conf.get('batch_size'):
            self.batch_size = 20
        else:
            if not isinstance(conf['batch_size'], int) or conf['batch_size'] < 1:
                raise ValueError(f'{conf_file}: batch_size: expected a positive integer.')
            self.batch_size = conf['batch_size']


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
    

    def ft_init(self) -> dict:

        height = len(self._labels)
        width = self.Xn_train.shape[1]
        params = {}


        match self.weights_init:

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

        if self.activation == 'sigmoid':
            activations['A'] = 1 / (1 + numpy.exp(-Z))

        return activations
    
    def ft_gradients(self, X, Y, activations) -> dict:

        m = X.shape[0]
        gradients = {}

        if self.loss == 'BCE':
            gradients['dW'] = (1 / m) * numpy.dot(activations['A'] - Y, X)
            gradients['db'] = (1 / m) * numpy.sum(activations['A'] - Y, axis=1, keepdims=True)

        return gradients
    
    def ft_update(self, gradients, params) -> dict:

        params['W'] -= self.learning_rate * gradients['dW']
        params['b'] -= self.learning_rate * gradients['db']

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

    def ft_compute_loss(self, X, Y, params):

        m = len(X)
        A = self.ft_activation(X, params)
        loss = (-1 / m) * numpy.sum(Y * numpy.log(A['A']) + (1 - Y) * numpy.log(1 - A['A']))
        return loss


    def train(self):

        X = self.Xn_train
        Y = numpy.array([[1 if x_label == label else 0 for x_label in self.Y_train] for label in self._labels])
        print(X.shape)
        print(Y.shape)

        params = self.ft_init()
        L = []
        for epoch in tqdm(range(self.epochs)):

            if self.optimization == 'GD':

                
                activations = self.ft_activation(X, params)
                gradients = self.ft_gradients(X, Y, activations)
                params = self.ft_update(gradients, params)
                # if epoch % 50 == 0:
                    # loss = self.ft_compute_loss()
                    # print(f'epoch({epoch}) - LOSS: {loss}')

                loss = self.ft_compute_loss(X, Y, params)
                L.append(loss)


            if self.optimization == 'SGD':

                r = int(random.random() * len(X))
                x = X[r].reshape(1, -1)
                y = Y[:, r].reshape(-1, 1)

                activations = self.ft_activation(x, params)
                gradients = self.ft_gradients(x, y, activations)
                params = self.ft_update(gradients, params)

                loss = self.ft_compute_loss(X, Y, params)
                L.append(loss)


        plt.plot(list(range(self.epochs)), L)
        plt.show()



        # ============== TESTING

        self.ft_testing(self.Xn_test, self.Y_test, params)


def main():

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--conf', default='conf.json')
        args = parser.parse_args()

        # dataset = pandas.read_csv(args.file)

        logreg = Logistic_Regression(args.conf)

        features = logreg._dataset.columns[6:]
        features_filtered = features.drop(['Arithmancy', 'Care of Magical Creatures'])

        logreg.setup('Hogwarts House', features_filtered)
        logreg.train()


    except BaseException as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
