import pandas as pd

def distancia_euclidiana(vet1, vet2):
  distancia = 0
  for i in range(len(vet1) -1):
    distancia += (vet1[i] - vet2[i])**2
  distancia = distancia**(1/2)
  return distancia

def retorna_vizinhos(base_treinamento, amostra_teste, k):
  distancias = []
  for i in base_treinamento:
    dist = distancia_euclidiana(amostra_teste, i)
    distancias.append((i, dist))
  distancias.sort(key=lambda tup:tup[1])
  vizinhos = []
  for i in range(k):
    vizinhos.append(distancias[i][0])
  return vizinhos

def classifica(base_treinamento, amostra_teste, k):
  vizinhos = retorna_vizinhos(base_treinamento, amostra_teste, k)
  rotulos = [v[-1] for v in vizinhos]
  print(rotulos)
  predicao = max(set(rotulos), key=rotulos.count)
  return predicao

def main():
  print("KNN")
  arquivo_entrada = 'winequality-white-normalized.csv'
  nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
  valores_independentes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
  valores_dependentes = 'quality'
  dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)
  dados_na_memoria.head()

  X=dados_na_memoria.
  Y=dados_na_memoria.
  treinamento = [[1,2,0],[2,3,0],[2,1,0],[2,7,1],[3,9,1],[2,4,1]]
  teste = [1,2,0]
  predicao = classifica(treinamento, teste, 3)
  print("resultado da Classificacao:")
  print('Rotulo esperado: %d\nRotulo Predicao: %d\n' % (teste[-1], predicao))


if __name__ == "__main__":
  main()
