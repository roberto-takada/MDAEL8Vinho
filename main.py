import pandas as pd
import numpy as np


def main():

    arquivo_entrada = 'winequality-all.csv'
    arquivo_saida = 'winequality-all-clean.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type']
    colunas_a_utilizar = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', "free sulfur dioxide", 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type']
    dados_na_memoria = pd.read_csv(arquivo_entrada, names=nomes_das_colunas, usecols=colunas_a_utilizar, na_values='?')

    print('PRIMEIRAS 10 LINHAS DO ARQUIVO \n')
    print(dados_na_memoria.head(10))
    print('\n')

    print('INFORMAÇÕES DOS DADOS: \n')
    print(dados_na_memoria.info())
    print('\n')

    print('DESCRIÇÃO DOS DADOS: \n')
    print(dados_na_memoria.describe())
    print('\n')

    print('VALORES FALTANTES: \n')
    print(dados_na_memoria.isnull().sum())
    print('\n')

    dados_na_memoria.to_numeric()
    mudanca = {'red': 1, 'white': 2}
    dados_na_memoria.replace({'red': mudanca, 'white': mudanca})

    print('PRIMEIRAS 10 LINHAS DO ARQUIVO \n')
    print(dados_na_memoria.head(10))
    print('\n')
    # dados_na_memoria.to_csv(arquivo_saida, header=False, index=False)


if __name__ == "__main__":
    main()
