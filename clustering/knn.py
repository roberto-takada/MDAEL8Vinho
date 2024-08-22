from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def knn_holdout(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

    knn = KNeighborsClassifier(n_neighbors=95)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    class_report = metrics.classification_report(y_test, y_pred)
    print(f'Relatório de classificação:\n{class_report}')
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
    plt.show()

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
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        score = cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))

    sns.lineplot(x=k_values, y=scores, marker='o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
    plt.show()
