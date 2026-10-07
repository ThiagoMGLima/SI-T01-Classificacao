# Decisões

Decisões de base da Tarefa 1 (Classificação), tomadas ao traçar o mapa em 2026-10-07. Os termos seguem o `GLOSSARY.md`. Decisões tomadas depois ficam nos tickets do mapa no GitHub.

## Escopo

1. **Destino**: o pacote de entrega pronto no repositório: relatório PDF com as 8 tabelas, códigos-fonte, `melhor_cart.joblib` e `melhor_rn.joblib`. A execução acontece dentro do mapa (tickets `task` rodam código e produzem resultados).
2. **Fora do escopo**: o envio no Moodle, feito manualmente pelo aluno.
3. **Condução**: trabalho individual; tickets atribuídos a @ThiagoMGLima. Sem prazo definido.

## Código e ambiente

4. **Formato**: projeto `uv` (`pyproject.toml` com scikit-learn, pandas, matplotlib, joblib) e scripts `.py` em `src/`, um por etapa (gerar dataset, treinar CART, treinar RN, teste cego, gerar relatório). Cada script salva seus números em `resultados/`; o relatório é montado a partir deles. Nomes e comentários em português.
5. **Fontes externas versionadas**: `gerar_dados_vitimas.py` copiado sem modificação para `vendor/victsim3/` (com crédito ao VictSim3) e importado; teste cego em `dados/teste_cego_1300v.csv`, lido **somente** pelo script do teste cego.
6. **Git**: commit e push direto na `main` ao fim de cada ticket, incluindo os `.joblib`.

## Dados e validação

7. **Dataset de treino/validação**: replica a configuração que gerou o teste cego (o `main` do gerador): proporção 100:390:405:405 escalada para 10.000 vítimas (769 verdes, 3000 amarelas, 3115 vermelhas, 3116 pretas), idade média 40, desvio 25, ruído 0,05, semente 42.
   _Por quê_: o padrão da função (2500 por classe) teria distribuição diferente da do teste cego.
8. **Validação cruzada**: k = 5 com `StratifiedKFold` (embaralhado, semente 42).
   _Por quê_: a classe verde é só ~8% das vítimas; sem estratificar, os folds ficam desbalanceados.
9. **DPA**: desvio padrão populacional (`ddof=0`) em todas as tabelas, inclusive a da idade.
   _Por quê_: o enunciado chama de "amostral", mas os números do exemplo (0.01074, 0.01185) só batem com `ddof=0`, que é o que o notebook do professor usa. Com 10.000 idades, a diferença só aparece na 4ª casa decimal.

## Modelos

10. **Critério de melhor CART / melhor RN** (o do notebook do professor), aplicado entre U, E e O de cada modelo: empate técnico entre quem tiver F1 macro médio de validação a até 0,005 do maior; desempate, nesta ordem, por menor DPA de validação, menor média da diferença treino-validação, maior F1 macro médio de validação.
11. **RN**: `MLPClassifier` do scikit-learn dentro de um `Pipeline` com `StandardScaler`, para que a padronização seja ajustada só no fold de treino. O CART não usa padronização.
12. **U/E/O**: escolhidas em tickets próprios (um para o CART, outro para a RN), a partir de varreduras exploratórias de F1 de treino × validação conforme a complexidade.

## Relatório

13. **Geração**: o PDF é gerado automaticamente a partir de `resultados/`, contendo **somente** as 8 tabelas do enunciado. A ferramenta é decidida em ticket próprio.
