import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def main():

  arquivo_entrada = 'winequality-white-normalized.csv'
  arquivo_saida = 'winequality-white-classificado.csv'
  nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
  colunas_a_utilizar = ['quality']
  dados_na_memoria = pd.read_csv(arquivo_entrada, names=nomes_das_colunas, usecols=colunas_a_utilizar, na_values='?')



  dados_na_memoria.to_csv(arquivo_saida, header=False, index=False)


if __name__ == "__main__":
    main()
