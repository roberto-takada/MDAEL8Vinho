from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from preprocessing.print_dados import print_dados


def minmax(dados, valores_independentes, valores_dependentes, arquivo_saida):
    x = dados.loc[:, valores_independentes].values
    y = dados.loc[:, [valores_dependentes]].values
    x_minmax = MinMaxScaler().fit_transform(x)
    dataframe_normalize_minmax = pd.DataFrame(data=x_minmax, columns=valores_independentes)
    dataframe_variaveis_dependentes = pd.DataFrame(data=y, columns=[valores_dependentes])
    dataframe_normalize_minmax = pd.concat([dataframe_normalize_minmax, dataframe_variaveis_dependentes], axis=1)
    print_dados(dataframe_normalize_minmax)
    dataframe_normalize_minmax.to_csv(arquivo_saida, header=False, index=False)
