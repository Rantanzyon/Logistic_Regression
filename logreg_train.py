import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def splitted_data(dataset: pd.DataFrame) -> tuple:

    index = int(len(dataset) * 0.8)
    dataset_train = dataset[:index]
    dataset_test = dataset[index:]

    # dataset_train = dataset[index:]
    # dataset_test = dataset[:index]
    return (dataset_train, dataset_test)

def mean(dataset: pd.DataFrame) -> dict:

    all_mean = {}
    for feature in dataset.columns[6:]:
        column = [e for e in dataset[feature] if e == e]
        mean = sum(column) / len(column)
        all_mean[feature] = mean
    return all_mean

def std(dataset: pd.DataFrame) -> dict:

    all_std = {}
    for feature in dataset.columns[6:]:
        column = [e for e in dataset[feature] if e == e]
        mean = sum(column) / len(column)
        var = sum([(e - mean) ** 2 for e in column]) / (len(column) - 1)
        std = var ** 0.5
        all_std[feature] = std
    return all_std

def normalize(dataset: pd.DataFrame, mean: dict, std: dict) -> np.array:

    features = dataset.columns[6:]
    mean = np.array(list(mean.values()))
    std = np.array(list(std.values()))
    dataset = np.array(dataset[features])
    dataset_norm = (dataset - mean) / std
    return dataset_norm

def training_logreg(X: np.array, dataset: pd.DataFrame) -> tuple:

    Ws = []
    bs = []
    for house in ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']:

        epochs = 1000
        lr = 0.1
        b = 0
        m = len(X)
        W = np.array([0.0] * len(X[0]))
        Y = np.array([1 if label == house else 0 for label in dataset['Hogwarts House']])

        for _ in range(epochs):

            Z = X.dot(W) + b
            A = 1 / (1 + np.exp(-Z))

            grad_w = (1 / m) * X.T.dot(A - Y)
            grad_b = (1 / m) * np.sum((A - Y))

            W -= lr * grad_w
            b -= lr * grad_b

        Ws.append(W)
        bs.append(b)

    Ws = np.array(Ws)
    bs = np.array(bs)

    return (Ws, bs)

def testing_logreg(W: np.array, b: np.array, dataset: pd.DataFrame, mean: dict, std: dict) -> float:

    houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
    dataset = dataset.fillna(value=mean)
    X = normalize(dataset, mean, std)
    Z = X.dot(W.T) + b
    A = 1 / (1 + np.exp(-Z))


    count = 0
    for i, stud in enumerate(A):
        best = 0
        for index, prob in enumerate(stud):
            if prob > best:
                house = houses[index]
                best = prob

        if house == dataset['Hogwarts House'].iloc[i]:
            count += 1

    accuracy = count * 100 / len(X)
    return accuracy


import json
def create_json_file(W, b, mean, std, dataset):

    houses = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']

    data_json = dict.fromkeys(houses)
    for i, house in enumerate(data_json.keys()):

        data_json[house] = dict.fromkeys(dataset.columns[6:])
        for j, feature in enumerate(data_json[house].keys()):
            data_json[house][feature] = {}
            data_json[house][feature]['mean'] = mean[feature]
            data_json[house][feature]['std'] = std[feature]
            data_json[house][feature]['w'] = W[i][j]
            data_json[house][feature]['b'] = b[i]
        # data_json[house]['b'] = b[i]


    json_formatted_string = json.dumps(data_json, indent=4)

    with open('datajson.json', 'w') as file:
        file.write(json_formatted_string)


def logistic_regression(dataset: pd.DataFrame):

    # delete useless columns based on histogram observation
    dataset = dataset.drop(columns=['Arithmancy', 'Care of Magical Creatures'])

    # split dataset 80/20
    # dataset_train, dataset_test = splitted_data(dataset)
    dataset_train = dataset.sample(frac=0.8)
    dataset_test = dataset.drop(dataset_train.index)

    # calculate mean et std
    mean_train = mean(dataset_train)
    std_train = std(dataset_train)

    # fill nan with mean
    dataset_train = dataset_train.fillna(value=mean_train)

    # normalize (x - mean) / std
    X_train_norm = normalize(dataset_train, mean_train, std_train)

    W, b = training_logreg(X_train_norm, dataset_train)
    accuracy = testing_logreg(W, b, dataset_test, mean_train, std_train)
    print(f'Accuracy: {accuracy:.2f}%')

    create_json_file(W, b, mean_train, std_train, dataset)






dataset = pd.read_csv('./datasets/dataset_train.csv')
logistic_regression(dataset)