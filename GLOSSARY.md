# Triagem de Vítimas

Classificar a cor de triagem START de vítimas de catástrofes a partir de sinais vitais e lesões, comparando um classificador CART com uma rede neural (Tarefa 1 de Sistemas Inteligentes 1, UTFPR).

## Vítimas e triagem

**Vítima**:
Pessoa atingida por catástrofe natural, desastre ou grande acidente, descrita pelas características de entrada.
_Avoid_: paciente

**Característica de entrada**:
Uma das 10 variáveis permitidas como entrada: idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim. gcs, avpu, sobr e tri nunca são características de entrada.
_Avoid_: feature, atributo

**Classe de triagem**:
A cor do protocolo START atribuída à vítima (variável `tri`): verde (0), amarelo (1), vermelho (2), preto (3). É a saída a ser predita.
_Avoid_: rótulo, gravidade, label

## Datasets

**Dataset de treino/validação**:
As 10.000 vítimas geradas por nós, usadas na validação cruzada e no retreino.
_Avoid_: dataset de treino (sozinho), base

**Teste cego**:
As 1.300 vítimas do VictSim3 (`1300v`), usadas uma única vez para comparar o melhor CART com a melhor RN.
_Avoid_: teste, holdout, validação

**Vazamento de dados**:
Qualquer uso do teste cego antes da etapa de teste cego; invalida a comparação final.
_Avoid_: data leakage

## Modelos e avaliação

**RN**:
Rede neural perceptron multicamadas (MLP) treinada para prever a classe de triagem.
_Avoid_: MLP, NN, rede

**Hiperparametrização**:
Um conjunto de valores de hiperparâmetros de um modelo. Cada modelo (CART e RN) tem exatamente três: U, E e O.
_Avoid_: parametrização, configuração

**U / E / O**:
As três hiperparametrizações de um modelo: subajustada (U), equilibrada (E) e sobreajustada (O).
_Avoid_: fraca/média/forte

**F1 macro médio**:
Média, sobre os k folds, do F1 macro de cada fold (média simples do F1 das quatro classes de triagem). Calculado separadamente para treino e validação.
_Avoid_: F1 (sozinho), acurácia

**DPA**:
Desvio padrão dos valores por fold, calculado dividindo por k (populacional), para reproduzir o exemplo do enunciado.
_Avoid_: desvio padrão amostral (o enunciado usa o nome, mas não a conta)

**Diferença treino-validação**:
Diferença absoluta entre o F1 macro de treino e o de validação em um fold; a média sobre os folds indica sobreajuste.
_Avoid_: viés (ambíguo com o viés do modelo)

**Melhor CART / Melhor RN**:
A hiperparametrização, entre U, E e O, vencedora do critério de seleção registrado em `decisoes.md`.
_Avoid_: melhor modelo (sem dizer qual)

**Retreino**:
Ajuste da hiperparametrização vencedora com todo o dataset de treino/validação, sem validação cruzada, antes do teste cego.
_Avoid_: fit final
