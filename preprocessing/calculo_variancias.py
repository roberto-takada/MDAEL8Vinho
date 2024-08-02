import pandas as pd
from preprocessing.print_graficos import print_grafico_histograma


def calculo_variancias(dados, opcao):
    desvio_padrao = dados.std()
    print("DESVIO PADRAO: \n")
    print(desvio_padrao)
    print("\n\n")
    variancia = pow(desvio_padrao, 2)
    print("VARIANCIA: \n")
    print(variancia)
    print("\n\n")
    media = dados.mean()
    coeficiente_de_variacao = (desvio_padrao / media) * 100
    print("COEFICIENTE DE VARIAÇÃO: \n")
    print(coeficiente_de_variacao)
    print("\n\n")
    menor_valor = dados.min()
    maior_valor = dados.max()
    amplitude = maior_valor - menor_valor
    print("AMPLITUDE: \n")
    print(amplitude)
    print("\n\n")

    print_grafico_histograma(dados, opcao)
    intervalos = [3, 5, 7, 9]
    dados['Intervalo de Classe'] = pd.cut(dados['quality'], intervalos)
    tabela_de_frequencia = dados['Intervalo de Classe'].value_counts(sort=False)
    tabela_de_frequencia = tabela_de_frequencia.sort_index()
    tabela_de_frequencia_relativa = tabela_de_frequencia / tabela_de_frequencia.sum()
    tabela_de_frequencia_acumulada = tabela_de_frequencia / tabela_de_frequencia.cumsum()
    tabela_final = pd.DataFrame(
        {'Frequencia': tabela_de_frequencia, 'Frequencia Relativa': tabela_de_frequencia_relativa,
         'Frequencia Acumulada': tabela_de_frequencia_acumulada})
    print(tabela_final)
