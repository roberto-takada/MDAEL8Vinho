import pandas as pd

from classification.decision_tree import decision_tree
from classification.random_forest import random_forest
from clustering.knn import knn_crossvalidation, knn_holdout
from classification.RNA import RNA
from classification.Xgboost import Xgboost
from preprocessing.calculo_variancias import calculo_variancias
from preprocessing.normalizacao_minmax import minmax
from preprocessing.normalizacao_zscore import z_score
from preprocessing.print_dados import print_dados
from preprocessing.print_graficos import print_matriz_correlacao, print_grafico_2variaveis
from preprocessing.projecao_pca import projecao_pca
from clustering.svm import svm_sklearn, svm_sklearn_pca
from clustering.Kmeans import Kmeans


def main():
    # COMENTAR AQUILO QUE NÃO FOR USAR
    # Lembrar de mudar as variáveis para vinho vermelho ou vinho branco
    arquivo_entrada = './database/winequality-white-clean.csv'
    arquivo_entrada_normalizado = './database/winequality-red-normalized.csv'
    # arquivo_saida_normalizado = './database/winequality-white-normalized.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                         'quality']
    valores_independentes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    valores_dependentes = 'quality'
    variavel1 = 'fixed acidity'
    variavel2 = 'free sulfur dioxide'

    dados = pd.read_csv(arquivo_entrada_normalizado, names=nomes_das_colunas, usecols=nomes_das_colunas, na_values='?')
    # print_dados(dados)
    # print_grafico_2variaveis(dados, variavel1, variavel2)

    # Calculo das Variancias
    # calculo_variancias(dados, 2)
    # dados = pd.read_csv(arquivo_entrada, names=nomes_das_colunas, usecols=nomes_das_colunas, na_values='?')
    # print_matriz_correlacao(dados)

    # NORMALIZAÇÃO
    # minmax(dados, valores_independentes, valores_dependentes, arquivo_saida_normalizado)
    # z_score(dados, valores_independentes, valores_dependentes, arquivo_saida_normalizado)

    # PROJECAO PCA 2 E 3 COMPONENTES
    # Mudar o 0 para 1 se precisar da projeção PCA para 3 componentes
    # Carregar o csv já normalizado na variavel dados
    # projecao_pca(dados, valores_independentes, valores_dependentes, 0)

    # svm_sklearn(dados, valores_independentes)
    # svm_sklearn_pca(dados, valores_independentes)
    # knn_holdout(dados)
    # knn_crossvalidation(dados)
    # decision_tree(dados)
    # random_forest(dados)
    # regressao(dados)
    # RNA(dados)
    # Xgboost(dados)
    # Kmeans(dados)

if __name__ == "__main__":
    main()
