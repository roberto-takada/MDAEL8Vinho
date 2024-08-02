import pandas as pd


def print_dados(dados):

    print('PRIMEIRAS 10 LINHAS DO ARQUIVO \n')
    print(dados.head(10))
    print('\n')

    print('INFORMACOES DOS DADOS: \n')
    print(dados.info())
    print('\n')

    print('DESCRICAO DOS DADOS: \n')
    print(dados.describe())
    print('\n')


