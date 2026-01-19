import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('./datasets/dataset_train.csv')
features = data.columns[6:]
houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']

# Remplacer les NaN par la moyenne
data[features] = data[features].fillna(data[features].mean())

# Split train/test 80/20
X = data[features].to_numpy()
y = data['Hogwarts House'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25, stratify=y
)

# Normalisation
mean_train = X_train.mean(axis=0)
std_train = X_train.std(axis=0)

X_train = (X_train - mean_train) / std_train
X_test  = (X_test  - mean_train) / std_train

# ================================
epochs = 500
lr = 0.05
m = len(X_train)

W_all = []
b_all = []

for house in houses:
    W = np.zeros(X_train.shape[1])
    b = 0
    Y = np.array([1 if label == house else 0 for label in y_train])

    for epoch in range(epochs):
        Z = X_train.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        grad_W = (1 / m) * X_train.T.dot(A - Y)
        grad_b = (1 / m) * np.sum(A - Y)
        W -= lr * grad_W
        b -= lr * grad_b

    W_all.append(W)
    b_all.append(b)

W_all = np.array(W_all)   # shape (4, 13)
b_all = np.array(b_all)   # shape (4,)

# ================================

Z_test = X_test.dot(W_all.T) + b_all
A_test = 1 / (1 + np.exp(-Z_test))
pred = np.array(houses)[np.argmax(A_test, axis=1)]
# print(pred[:10])

print("Accuracy One-vs-All vectorisée :", accuracy_score(y_test, pred))

# ================================

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
sk_pred = clf.predict(X_test)
# print(sk_pred[:10])


print("Accuracy sklearn LogisticRegression :", accuracy_score(y_test, sk_pred))

# ================================

data = pd.read_csv('./datasets/dataset_test.csv')
features = data.columns[6:]
houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']

X = data[features].fillna(data[features].mean())
X = X.to_numpy()

X = (X - mean_train) / std_train

Z = X.dot(W_all.T) + b_all
A = 1 / (1 + np.exp(-Z))
pred = np.array(houses)[np.argmax(A, axis=1)]

print(pred[:10])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# print(X)
sk_pred = clf.predict(X)
print(sk_pred[:10])

with open('houses.csv', 'w') as file:
    
    file.write('Index,Hogwarts House\n')
    for index, house in enumerate(pred):

        file.write(f'{index},{house}\n')

#======================================================

import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/dataset_test.csv')
data = data[features].to_numpy()


data_train = pd.read_csv('./datasets/dataset_train.csv')



for j, stud in enumerate(data):

    print(f'pred for stud {j}: {pred[j]}')
    fig, axes = plt.subplots(4,4,figsize=(16,12))
    axes = axes.flatten()
    for i, feature in enumerate(features):

        note = stud[i]
        for house in houses:

            feature_data = data_train[data_train['Hogwarts House'] == house][feature].astype(float)
            axes[i].hist(feature_data, bins=20, density=True, label=house, alpha=0.5)

        axes[i].scatter([note],[0], color='black')
        axes[i].set_title(feature)


    fig.legend(houses)
    for j in range(len(features), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    
