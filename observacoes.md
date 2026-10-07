# Observações

Decisões de base da Tarefa 1 (Classificação), tomadas ao traçar o mapa em 2026-10-07, e observações sobre o que cada etapa produziu. Os termos seguem o `GLOSSARY.md`. Decisões tomadas depois ficam nos tickets do mapa no GitHub.

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

## Dataset de treino/validação

Observado ao gerar o dataset (ticket "Gerar o dataset de treino/validação (10.000 vítimas)", 2026-10-07).

14. **Contagens por classe de triagem após o ruído**: o gerador recebe 769/3000/3115/3116 vítimas por classe, mas troca a classe de triagem por uma vizinha em ~5% delas. A Tabela 1 traz as contagens após essa troca: 805/2966/3179/3050. `resultados/tabela1_dataset.json` guarda os dois conjuntos (`parametros_gerador` e `tabela_1`).
15. **Idade abaixo dos parâmetros**: média 39,92 e DPA 23,20, contra os 40 e 25 passados ao gerador, porque ele limita a idade a [1, 90] e trunca para inteiro. A Tabela 1 usa os valores medidos (item 9).
16. **Efeitos colaterais do gerador**: ao ser importado, ele cria `./datasets/vict/1300v/`, e sempre salva no caminho fixo `OUTPUT_CSV`. `src/gerar_dataset.py` contorna os dois sem editar o arquivo: importa o gerador a partir de uma pasta temporária e reaponta `OUTPUT_CSV` para `dados/`.
17. **Colunas do CSV**: `dados/treino_validacao_10000v.csv` tem as 14 colunas do gerador; os scripts de treino precisam manter só as 10 características de entrada.

## Validação cruzada e seleção

Observado ao montar o módulo de validação cruzada (ticket "Montar o módulo de validação cruzada e seleção do melhor modelo", 2026-10-07).

18. **Módulo comum**: `src/validacao_cruzada.py`. Os scripts de `src/`, rodados com `uv run python src/<script>.py`, importam com `from validacao_cruzada import ...`. Funções: `carregar_dataset()` (X com as 10 características de entrada, y = `tri`; resolve o item 17), `avaliar(estimador, hiperparametrizacao, X, y)`, `escolher_melhor({"U": ..., "E": ..., "O": ...})` (item 10; devolve o rótulo), `formatar(...)` (célula `0.88212 (0.01074)` das Tabelas 3, 5 e 6) e `montar(estimador, hiperparametrizacao)`. `uv run python src/validacao_cruzada.py` confere o exemplo do enunciado.
19. **Formato do resultado**: `avaliar` devolve um dicionário serializável em JSON: `hiperparametrizacao` e `treino`/`validacao`/`diferenca`, cada um com `por_fold`, `media` e `dpa` (`ddof=0`). A diferença treino-validação é absoluta, fold a fold. Os scripts de treino podem gravá-lo direto em `resultados/`.
20. **Semente nos modelos**: `montar` põe 42 em todo `random_state` deixado em `None`, inclusive dentro do Pipeline (`mlp__random_state`). Na RN, os hiperparâmetros levam o prefixo do passo do Pipeline (ex.: `mlp__hidden_layer_sizes`). O retreino deve montar o modelo com `montar` para reproduzir o da validação.
21. **Mesmos folds para todos**: toda hiperparametrização, de CART e de RN, é avaliada na mesma divisão: 8.000 vítimas de treino e 2.000 de validação por fold, com 161 verdes em cada fold de validação.
22. **Classe nunca predita**: entra com F1 0 no F1 macro (`zero_division=0`, sem aviso), o que pesa nas U muito simples. Um classificador que sempre prediz a classe mais frequente tem F1 macro médio de validação 0.12061.
23. **Pontos de referência (não são U/E/O)**: a árvore com hiperparâmetros padrão (sem limite de profundidade) dá treino 1.00000 (0.00000), validação 0.87844 (0.00405) e diferença 0.12156 (0.00405), em 0,2 s por avaliação. A RN padrão (`MLPClassifier()` no Pipeline) dá treino 0.94548 (0.00143), validação 0.94386 (0.00620) e diferença 0.00624 (0.00467), em ~12 s por avaliação, e não converge em 200 iterações (`ConvergenceWarning` em todos os folds).

## Relatório PDF

Decidido com o usuário a partir de protótipos (ticket "Definir como o relatório PDF é gerado", 2026-10-07).

24. **Ferramenta**: Typst, pelo pacote PyPI `typst`, que traz o compilador embutido (a máquina não tem pandoc nem LaTeX). Um template `.typ` faz o layout; o script Python lê `resultados/`, formata os números e os passa ao template como JSON (`sys.inputs`). Ponto de partida: branch descartável `prototipo/relatorio-pdf` (commit 08d050c), arquivos `src/prototipo_relatorio/candidato_a_typst/relatorio.typ`, `candidato_a_typst.py` e `comum.py`.
25. **Fontes e cores**: o template usa fontes do sistema com as métricas das do enunciado: Carlito (Calibri) no texto e Liberation Mono (Courier New) nas matrizes de confusão. Sem elas, o Typst cai na fonte padrão. O f̄1 sai na fonte matemática do Typst. As cores dos cabeçalhos foram tiradas do enunciado: #E5DFEC (Tabela 1), #DAEEF3 (2 e 3), #FDE9D9 (4 e 5), #B2A1C7 (6 a 8).
26. **Conteúdo das tabelas**: o rótulo "Desvio padrão amostral da idade" fica como no enunciado, embora o valor seja populacional (item 9); o separador decimal é o ponto; a linha "ruído" mostra o valor usado (0.05); não há texto fora das 8 tabelas (item 13).
27. **Exibição de `max_depth=None`**: ficou em aberto aqui e foi decidida no ticket das U/E/O do CART: a Tabela 2 mostra o literal `None` (item 31).

## U/E/O do CART

Decidido com o usuário a partir de uma varredura exploratória (ticket "Escolher as hiperparametrizações U/E/O do CART", 2026-10-07). A varredura está no branch descartável `prototipo/varredura-cart` (commit 37b30b5): `src/prototipo_varredura_cart.py`, com gráficos e tabelas em `resultados/prototipo_varredura_cart/`. São 454 hiperparametrizações avaliadas com `avaliar`.

28. **U/E/O escolhidas** (Tabela 2): entropy nas três. U: `min_samples_leaf=1`, `max_depth=2`. E: `min_samples_leaf=8`, `max_depth=8`. O: `min_samples_leaf=1`, `max_depth=None`.
    _Por quê_: um eixo só de complexidade, com o melhor criterion (item 33). E fica no platô do pico (item 34), com a menor diferença treino-validação entre os 10 melhores. U e O ficam nas duas pontas da mesma curva. Dois conjuntos foram descartados. Gini nas três deixaria O igual à árvore padrão (item 23), mas o E sairia 0,006 abaixo e com DPA maior. Variar só `min_samples_leaf` faria o U depender de a árvore deixar de predizer verde (item 35).
29. **Tabela 3 e melhor CART**: treino / validação / média das difs. U: 0.57983 (0.00164) / 0.57981 (0.00656) / 0.00660 (0.00487). E: 0.94194 (0.00154) / 0.93766 (0.00777) / 0.00718 (0.00728). O: 1.00000 (0.00000) / 0.87859 (0.00686) / 0.12141 (0.00686). O melhor CART é **E**, sem empate técnico: a validação de E fica 0,059 acima da de O. No retreino, montar com `montar(DecisionTreeClassifier(), {"min_samples_leaf": 8, "max_depth": 8, "criterion": "entropy"})` (item 20).
30. **Onde ficam as Tabelas 2 e 3**: `src/treinar_cart.py` grava `resultados/tabelas2_3_cart.json` com três chaves. `tabela_2` mapeia U/E/O para a hiperparametrização (`min_samples_leaf`, `max_depth`, `criterion`). `tabela_3` mapeia U/E/O para `treino`/`validacao`/`diferenca`, cada um com `por_fold`, `media` e `dpa`, como em `avaliar`. `melhor_cart` guarda o rótulo. A Tabela 6 sai de `tabela_3[melhor_cart]`.
31. **`max_depth=None` na Tabela 2**: a profundidade ilimitada aparece como o literal `None`, o valor do scikit-learn (decidido com o usuário; resolve o item 27). O JSON guarda `null`, e a conversão para "None" fica com o script do relatório.
32. **Sobreajuste a partir de `max_depth` ~7**: com `min_samples_leaf=1`, a validação tem o pico em `max_depth=8` com entropy (0.93474) e em 12 com gini (0.92142). Depois cai até ~0,878 sem limite, enquanto o treino vai a 1.00000. A diferença treino-validação cresce sem parar a partir de ~7 (passa de 0,01 em 7 com entropy) e chega a 0,12 sem limite.
33. **entropy > gini até o pico**: com `max_depth` de 3 a 11, a validação com entropy fica 0,006 a 0,045 maior e o DPA de validação é menor (~0,007 contra ~0,012). Com mais profundidade, as duas ficam parecidas. O melhor ponto de cada uma em toda a varredura: entropy 0.93790, gini 0.93147.
34. **Platô do `min_samples_leaf`**: `min_samples_leaf` regulariza. Sem limite de profundidade, ir de 1 para 10 leva a validação de 0.87859 para 0.93552 e a diferença de 0.12141 para 0.00787. Acima de ~15, a validação cai aos poucos (0.85066 com 300). O pico é um platô: entropy com `max_depth` 8–12 e `min_samples_leaf` 8–10 dá validação de 0.9372 a 0.9379, uma diferença menor que um DPA.
35. **U rasa e a classe verde**: com `max_depth=2`, a árvore nunca prediz verde, e essa classe entra com F1 0 (item 22). Com `max_depth=1`, só prediz amarelo e preto (0.37). Com `max_depth=3`, já prediz verde e chega a 0.85 com entropy, alto demais para ser U. Com entropy e `min_samples_leaf` ≥ 750, a árvore também deixa de predizer verde (cada fold de treino tem 644 verdes) e o F1 cai para 0.61.

## U/E/O da RN

Decidido com o usuário a partir de uma varredura exploratória (ticket "Escolher as hiperparametrizações U/E/O da RN", 2026-10-07). A varredura está no branch descartável `prototipo/varredura-rn` (commit 77dc8ee): `src/prototipo_varredura_rn/`, com gráficos `saida/rn_*.png` e a tabela `saida/varredura_rn.md`. São 79 hiperparametrizações avaliadas com `avaliar`, varrendo topologia, função de ativação, alpha, solver/learning rate e max_iter.

36. **U/E/O escolhidas** (Tabela 4): `max_iter=2000` nas três. U: topologia [2], logistic, sgd, learning rate 0.001, alpha 0.0001. E: [64], relu, adam, learning rate 0.001, alpha 0.0001. O: [128 128], relu, lbfgs, alpha 0.
    _Por quê_: capacidade e otimização crescem juntas, e as quatro linhas do enunciado variam. As três ficam bem separadas: U baixa no treino e na validação; E com o maior F1 de validação entre as RN que convergem e DPA de validação baixo; O decorando o treino como a O do CART (item 29). Dois conjuntos foram descartados. Só adam (U [1] logistic, E [32 16], O [128 128 128] tanh com alpha 0) preencheria a linha Learning rate nas três, mas U (0.839) e O (diferença 0.081) ficariam menos extremas. A mesma topologia [64 64] com alpha 30 / 0.1 / 0 (e lbfgs na O) explicaria tudo por um botão só, mas a linha Topologia não variaria.
37. **Tabela 5 e melhor RN**: treino / validação / média das difs. U: 0.65897 (0.00153) / 0.65887 (0.00622) / 0.00579 (0.00510). E: 0.94599 (0.00152) / 0.94383 (0.00433) / 0.00446 (0.00420). O: 1.00000 (0.00000) / 0.88994 (0.01024) / 0.11006 (0.01024). A melhor RN é **E**, sem empate técnico: a validação de E fica 0,054 acima da de O. No retreino, montar com `montar(RN, HIPERPARAMETRIZACOES["E"])`, importando os dois de `treinar_rn` (o Pipeline com os passos `padronizacao` e `mlp`; os hiperparâmetros levam o prefixo `mlp__`, item 20).
38. **Onde ficam as Tabelas 4 e 5**: `src/treinar_rn.py` grava `resultados/tabelas4_5_rn.json` com quatro chaves. `tabela_4` mapeia U/E/O para `hidden_layer_sizes` (lista de neurônios por camada oculta), `activation`, `learning_rate_init`, `solver`, `alpha` e `max_iter`, sem o prefixo `mlp__`. As quatro primeiras são as linhas do enunciado; `alpha` e `max_iter` completam a hiperparametrização. `tabela_5` tem o mesmo formato da `tabela_3` do CART (item 30). `melhor_rn` guarda o rótulo, e a Tabela 6 sai de `tabela_5[melhor_rn]`. `execucao` mapeia U/E/O para `tempo_s` (a avaliação inteira) e `folds_sem_convergir` (folds com `ConvergenceWarning`).
39. **`max_iter=2000` para convergir**: com o padrão (200), a RN não converge (item 23). Com 2000, os 5 folds das três param pelo critério de tol, sem aviso. U usa 683–699 épocas, E 345–450 e O 704–973 iterações do lbfgs. `treinar_rn.py` não silencia os avisos: reemite cada aviso distinto com a contagem e grava `folds_sem_convergir`. Na varredura, o lbfgs com [32 16] e alpha 0.0001 não convergiu em 2000 iterações.
40. **1 thread de BLAS**: `treinar_rn.py` roda `avaliar` dentro de `threadpool_limits(1)` (threadpoolctl, dependência do scikit-learn). Com as threads padrão do BLAS (~14 dos 16 núcleos ocupados), a O passou de 10 min sem terminar. Com 1 thread, leva 141 s (U 16 s, E 17 s), com os mesmos números. O retreino deve usar `threadpool_limits(1)` também.
41. **Learning rate do lbfgs**: o lbfgs ignora `learning_rate_init`, então a linha Learning rate da O mostra "—" (decidido com o usuário). O JSON guarda `null`, e a conversão para "—" fica com o script do relatório, como no `max_depth=None` do CART (item 31).
42. **U e a classe verde**: a U nunca prediz verde, e essa classe entra com F1 0 (item 22), como a U do CART (item 35); o usuário aceitou. F1 de validação por classe da U: verde 0, amarelo 0.820, vermelho 0.874, preto 0.941. A U do conjunto descartado só com adam ([1] logistic, 0.839) predizia as quatro classes.
43. **Subajuste exige estrangular a RN**: o problema é fácil. Uma RN linear (`identity`, [32 16]) dá 0.911 de validação, [2] relu dá 0.903 e 5 épocas da RN padrão dão 0.895. Com alpha=30 ([2], [32 16] e [64 64]) ou [2] logistic + sgd, a RN nunca prediz verde e fica em 0.62–0.66. Com alpha 10–15, conforme a topologia, o F1 fica instável entre os folds (DPA de validação até 0,057). Com alpha=100 ou [1] relu (neurônio morto), tudo colapsa na classe mais frequente (0.12061).
44. **Sobreajuste vem de profundidade × largura com alpha≈0**: com uma camada só, até 256 neurônios, a diferença fica em até 0,013. Uma camada de 16 a 128 neurônios, ou [32 16], é o platô equilibrado (validação 0.941–0.944, diferença 0.004–0.008), assim como [64 64] com alpha 0.1–1. Com duas ou três camadas e adam até convergir, a diferença cresce com a largura: 0.010 em [32 32], 0.038–0.043 em [64 64], 0.055–0.069 de [64 64 64] e [128 128] para cima. Com tanh ([128 128 128], 0.081) ou mais paciência (`n_iter_no_change=50`, 0.093), vai mais longe. Só o lbfgs com alpha 0 decora o treino: F1 de treino 1.00000 e diferença 0.110–0.121 em [64 64], [128 128] e [256 256].
