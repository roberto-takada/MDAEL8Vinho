from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def knn_holdout(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def knn_crossvalidation(dados):
    k_values = [i for i in range(1, 100)]
    print('Crossvalidation')
    scores = []
    X = dados.drop('quality', axis=1)
    y = dados['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for k in k_values:

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        score = cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))

    sns.lineplot(x=k_values, y=scores, marker='o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.show()