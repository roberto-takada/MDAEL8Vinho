from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np


def svm_sklearn(dados, variaveis_independentes):
    X_train, X_test, y_train, y_test = train_test_split(dados[variaveis_independentes], dados['quality'], test_size=0.3,
                                                        random_state=109)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    clf_svm = SVC(random_state=42, C=100, gamma=1)
    clf_svm.fit(X_train_scaled, y_train)
    scores = cross_val_score(clf_svm, dados[variaveis_independentes], dados['quality'], cv=5)

    # calculate overall accuracy
    y_pred = clf_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2%}')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    class_names = ['3', '4', '5', '6', '7', '8', '9']
    disp = ConfusionMatrixDisplay.from_estimator(
        clf_svm,
        X_test_scaled,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues)
    plt.show()


def svm_sklearn_pca(dados, variaveis_independentes):
    x = dados.loc[:, variaveis_independentes].values
    pca = PCA()
    componentes_principais = pca.fit_transform(x)
    colunas = ['ComponentePrincipal1', 'ComponentePrincipal2']
    dataframe_principal = pd.DataFrame(data=componentes_principais[:, 0:2],
                                       columns=colunas)
    dataframe_final = pd.concat([dataframe_principal, dados[['quality']]], axis=1)
    y = np.ravel(dataframe_final['quality'])
    X_train, X_test, y_train, y_test = train_test_split(dataframe_final[colunas], y, test_size=0.3,
                                                        random_state=109)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    clf_svm = SVC(random_state=42, C=100, gamma=1)
    clf_svm.fit(X_train_scaled, y_train)
    scores = cross_val_score(clf_svm, dataframe_final[colunas], y, cv=5)
    y_pred = clf_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2%}')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    class_names = ['3', '4', '5', '6', '7', '8', '9']
    disp = ConfusionMatrixDisplay.from_estimator(
        clf_svm,
        X_test_scaled,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues)
    plt.show()