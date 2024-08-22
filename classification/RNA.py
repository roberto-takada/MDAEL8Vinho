import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn

def RNA(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Criar e treinar o MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128,128,50), max_iter=500, activation='relu')
    mlp.fit(X_train, y_train)

    # Fazer previsões
    y_pred = mlp.predict(X_test)

    
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Avaliar o modelo
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))
    sn.heatmap(conf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
    plt.show()