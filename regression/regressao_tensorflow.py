import numpy as np


def regressao(dados):
    dados.head()
    dados['target'] = dados['quality']
