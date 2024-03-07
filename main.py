import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def main():

    arquivo_entrada = 'winequality-all.csv'
    arquivo_saida = 'winequality-all-clean.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type']
    colunas_a_utilizar = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type']
    dados_na_memoria = pd.read_csv(arquivo_entrada, names=nomes_das_colunas, usecols=colunas_a_utilizar, na_values='?')

    MostraInformacoesDoDataFrame(dados_na_memoria)

    print('VALORES FALTANTES: \n')
    print(dados_na_memoria.isnull().sum())
    print('\n')

    dados_na_memoria.to_csv(arquivo_saida, header=False, index=False)

    arquivo_entrada = 'winequality-all-clean.csv'
    arquivo_saida = 'winequality-all-normalized.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    valores_independentes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    valores_dependentes = 'quality'
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)

    x = dados_na_memoria.loc[:, valores_independentes].values
    y = dados_na_memoria.loc[:, [valores_dependentes]].values

    #Normalização com Z-score
    x_zscore = StandardScaler().fit_transform(x)
    dataframe_normalizado_zscore = pd.DataFrame(data = x_zscore, columns = valores_independentes)
    #dataframe_normalizado_zscore = pd.concat([dataframe_normalizado_zscore, dados_na_memoria[[valores_dependentes]]], axis = 1)
    MostraInformacoesDoDataFrame(dataframe_normalizado_zscore)

    #Normalização Min-Max
    x_minmax = MinMaxScaler().fit_transform(x)
    dataframe_normalizado_minmax = pd.DataFrame(data = x_minmax, columns = valores_independentes)
    #dataframe_normalizado_minmax = pd.concat([dataframe_normalizado_minmax, dados_na_memoria[[valores_dependentes]]], axis = 1)
    MostraInformacoesDoDataFrame(dataframe_normalizado_minmax)


def MostraInformacoesDoDataFrame(dados_na_memoria):
    print('PRIMEIRAS 10 LINHAS DO ARQUIVO \n')
    print(dados_na_memoria.head(10))
    print('\n')

    print('INFORMACOES DOS DADOS: \n')
    print(dados_na_memoria.info())
    print('\n')

    print('DESCRICAO DOS DADOS: \n')
    print(dados_na_memoria.describe())
    print('\n')


if __name__ == "__main__":
    main()
