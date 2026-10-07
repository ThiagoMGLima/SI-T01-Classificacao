"""Retreina o melhor CART e a melhor RN, salva os modelos e roda o teste cego (Tabelas 6, 7 e 8).

Etapa 5 do enunciado. Os rótulos do melhor CART e da melhor RN saem dos JSON das Tabelas 2 e 3
e 4 e 5 (`melhor_cart`, `melhor_rn`). Cada um é montado como na validação cruzada (montar,
semente 42, item 20 de observacoes.md), retreinado com as 10.000 vítimas do dataset de
treino/validação, sem validação cruzada, e salvo em modelos/melhor_cart.joblib e
modelos/melhor_rn.joblib. Só depois disso o teste cego é lido: este é o único script do projeto
que o lê (item 5), e as métricas saem dos modelos recarregados dos .joblib, os mesmos da entrega.
Os avisos de não convergência da RN não são silenciados.

Uso: uv run python src/teste_cego.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from treinar_cart import HIPERPARAMETRIZACOES as HIPERPARAMETRIZACOES_CART
from treinar_cart import TABELAS_2_3
from treinar_rn import HIPERPARAMETRIZACOES as HIPERPARAMETRIZACOES_RN
from treinar_rn import RN, TABELAS_4_5, tabela_4
from validacao_cruzada import CARACTERISTICAS, CLASSE, K_FOLDS, SEMENTE, carregar_dataset, formatar, montar

RAIZ = Path(__file__).resolve().parent.parent
TESTE_CEGO = RAIZ / "dados" / "teste_cego_1300v.csv"
MODELOS = RAIZ / "modelos"
MELHOR_CART = MODELOS / "melhor_cart.joblib"
MELHOR_RN = MODELOS / "melhor_rn.joblib"
TABELAS_6_7_8 = RAIZ / "resultados" / "tabelas6_7_8_teste_cego.json"

# Classes de triagem na ordem de tri (0 a 3) e as letras das matrizes de confusão da Tabela 8.
CORES = {0: "verde", 1: "amarelo", 2: "vermelho", 3: "preto"}
TRI = list(CORES)
LETRAS = ["G", "Y", "R", "B"]

LINHAS_TABELA_6 = {
    "treino": "TREINO f1 (DPA)",
    "validacao": "VALIDAÇÃO f1 (DPA)",
    "diferenca": "MÉDIA DAS DIFS. (DPA)",
}
LINHAS_TABELA_7 = {
    "precisao_macro": "MÉDIA PRECISÃO (macro)",
    "recall_macro": "MÉDIA RECALL (macro)",
    "f1_macro": "F1 SCORE (macro)",
    "acuracia": "ACURÁCIA",
}


def ler_json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def melhores():
    """Rótulos do melhor CART e da melhor RN e a Tabela 6, a partir dos JSON das Tabelas 2 a 5.

    A Tabela 6 copia `tabela_3[melhor_cart]` e `tabela_5[melhor_rn]`. Confere que as
    hiperparametrizações gravadas nesses JSON são as dos scripts de treino, para que o retreino
    use a mesma hiperparametrização que foi validada.
    """
    cart, rn = ler_json(TABELAS_2_3), ler_json(TABELAS_4_5)
    melhor_cart, melhor_rn = cart["melhor_cart"], rn["melhor_rn"]
    if cart["tabela_2"][melhor_cart] != HIPERPARAMETRIZACOES_CART[melhor_cart]:
        raise SystemExit(f"CART {melhor_cart}: a Tabela 2 do JSON difere de treinar_cart.py")
    if rn["tabela_4"][melhor_rn] != tabela_4()[melhor_rn]:
        raise SystemExit(f"RN {melhor_rn}: a Tabela 4 do JSON difere de treinar_rn.py")
    tabela_6 = {"cart": cart["tabela_3"][melhor_cart], "rn": rn["tabela_5"][melhor_rn]}
    return melhor_cart, melhor_rn, tabela_6


def estatisticas(valores):
    return {
        "por_fold": [float(v) for v in valores],
        "media": float(np.mean(valores)),
        "dpa": float(np.std(valores, ddof=0)),
    }


def f1_validacao_por_classe(estimador, hiperparametrizacao, X, y):
    """F1 de validação de cada classe de triagem, nos mesmos folds de avaliar (k=5, semente 42).

    Só usa o dataset de treino/validação; serve para comparar com o F1 por classe do teste cego.
    A média das quatro classes reproduz o F1 macro médio de validação das Tabelas 3 e 5
    (conferido em main). Roda com 1 thread de BLAS, como a validação da RN (item 40).
    """
    folds = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    por_fold = []
    with threadpool_limits(1):
        for treino, validacao in folds.split(X, y):
            modelo = montar(estimador, hiperparametrizacao).fit(X.iloc[treino], y.iloc[treino])
            predito = modelo.predict(X.iloc[validacao])
            por_fold.append(f1_score(y.iloc[validacao], predito, labels=TRI, average=None, zero_division=0))
    por_fold = np.array(por_fold)
    return {cor: estatisticas(por_fold[:, i]) for i, cor in enumerate(CORES.values())}


def retreinar_cart(hiperparametrizacao, X, y):
    """Melhor CART montado como na validação e ajustado com as 10.000 vítimas, sem validação cruzada."""
    modelo = montar(DecisionTreeClassifier(), hiperparametrizacao)
    inicio = time.perf_counter()
    modelo.fit(X, y)
    tempo = time.perf_counter() - inicio
    return modelo, {"tempo_s": tempo, "profundidade": int(modelo.get_depth()), "folhas": int(modelo.get_n_leaves())}


def retreinar_rn(hiperparametrizacao, X, y):
    """Melhor RN montada como na validação e ajustada com as 10.000 vítimas, sem validação cruzada.

    Roda com 1 thread de BLAS (item 40). O ConvergenceWarning não é silenciado: é reemitido, e o
    resultado registra se a RN convergiu e quantas épocas usou (`n_iter_` do MLPClassifier).
    """
    modelo = montar(RN, hiperparametrizacao)
    with threadpool_limits(1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        modelo.fit(X, y)
        tempo = time.perf_counter() - inicio
    for a in avisos:
        warnings.showwarning(a.message, a.category, a.filename, a.lineno)
    convergiu = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    return modelo, {"tempo_s": tempo, "epocas": int(modelo.named_steps["mlp"].n_iter_), "convergiu": convergiu}


def carregar_teste_cego():
    """Teste cego: X com as 10 características de entrada e y = classe de triagem. Lido só aqui."""
    df = pd.read_csv(TESTE_CEGO)
    return df[CARACTERISTICAS], df[CLASSE]


def avaliar_teste_cego(modelo, X, y):
    """Métricas da Tabela 7, matriz de confusão da Tabela 8 e F1 por classe de um modelo no teste cego.

    A matriz tem linhas = classe real e colunas = classe predita, na ordem G, Y, R, B (tri 0 a 3).
    """
    predito = modelo.predict(X)
    metricas = {
        "precisao_macro": float(precision_score(y, predito, labels=TRI, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predito, labels=TRI, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predito, labels=TRI, average="macro", zero_division=0)),
        "acuracia": float(accuracy_score(y, predito)),
    }
    matriz = confusion_matrix(y, predito, labels=TRI).tolist()
    f1_classes = f1_score(y, predito, labels=TRI, average=None, zero_division=0)
    return metricas, matriz, {cor: float(f) for cor, f in zip(CORES.values(), f1_classes)}


def imprimir_tabela(titulo, colunas, linhas):
    """Tabela em texto: `linhas` mapeia o nome da linha -> células de cada coluna."""
    print(f"\n{titulo}")
    print(f"{'MODELO':<26}" + "".join(f"{c:<30}" for c in colunas))
    for nome, celulas in linhas.items():
        print(f"{nome:<26}" + "".join(f"{str(c):<30}" for c in celulas))


def imprimir_matriz(titulo, matriz):
    print(f"\n{titulo} (linhas: real; colunas: predito)")
    print("    " + "".join(f"{letra:>6}" for letra in LETRAS))
    for letra, linha in zip(LETRAS, matriz):
        print(f"{letra:<4}" + "".join(f"{n:>6}" for n in linha))


def main():
    melhor_cart, melhor_rn, t6 = melhores()
    hiperparametrizacao = {"cart": HIPERPARAMETRIZACOES_CART[melhor_cart], "rn": HIPERPARAMETRIZACOES_RN[melhor_rn]}
    estimador = {"cart": DecisionTreeClassifier(), "rn": RN}
    X, y = carregar_dataset()

    # F1 de validação por classe, só com o dataset de treino/validação. Reproduzir o F1 macro
    # médio de validação do JSON confere que montar dá o mesmo modelo que foi validado.
    f1_validacao = {}
    for nome in estimador:
        f1_validacao[nome] = f1_validacao_por_classe(estimador[nome], hiperparametrizacao[nome], X, y)
        macro = np.mean([e["media"] for e in f1_validacao[nome].values()])
        if not np.isclose(macro, t6[nome]["validacao"]["media"], rtol=0, atol=1e-12):
            raise SystemExit(f"{nome}: F1 macro médio de validação recalculado ({macro}) difere do JSON")

    # Retreino com as 10.000 vítimas; os modelos são salvos antes de qualquer leitura do teste cego.
    cart, retreino_cart = retreinar_cart(hiperparametrizacao["cart"], X, y)
    rn, retreino_rn = retreinar_rn(hiperparametrizacao["rn"], X, y)
    MODELOS.mkdir(exist_ok=True)
    joblib.dump(cart, MELHOR_CART)
    joblib.dump(rn, MELHOR_RN)
    print(f"Melhor CART ({melhor_cart}) retreinado em {retreino_cart['tempo_s']:.2f} s: {retreino_cart}")
    print(f"Melhor RN ({melhor_rn}) retreinada em {retreino_rn['tempo_s']:.2f} s: {retreino_rn}")
    print(f"Modelos salvos em: {MELHOR_CART.relative_to(RAIZ)}, {MELHOR_RN.relative_to(RAIZ)}")

    # Teste cego: primeira e única leitura, com os modelos recarregados dos .joblib da entrega.
    X_cego, y_cego = carregar_teste_cego()
    t7, t8, f1_cego = {}, {}, {}
    for nome, caminho in [("cart", MELHOR_CART), ("rn", MELHOR_RN)]:
        t7[nome], t8[nome], f1_cego[nome] = avaliar_teste_cego(joblib.load(caminho), X_cego, y_cego)

    contagem = y_cego.value_counts()
    saida = {
        "melhor_cart": melhor_cart,
        "melhor_rn": melhor_rn,
        "tabela_6": t6,
        "tabela_7": t7,
        "tabela_8": {"ordem": LETRAS, "linhas": "real", "colunas": "predito", **t8},
        "f1_por_classe": {nome: {"validacao": f1_validacao[nome], "teste_cego": f1_cego[nome]} for nome in estimador},
        "teste_cego": {
            "vitimas": int(len(y_cego)),
            "vitimas_por_classe": {cor: int(contagem.get(tri, 0)) for tri, cor in CORES.items()},
        },
        "modelos": {"cart": str(MELHOR_CART.relative_to(RAIZ)), "rn": str(MELHOR_RN.relative_to(RAIZ))},
        "retreino": {"cart": retreino_cart, "rn": retreino_rn},
    }
    TABELAS_6_7_8.parent.mkdir(exist_ok=True)
    TABELAS_6_7_8.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    colunas = [f"Melhor CART ({melhor_cart})", f"Melhor RN ({melhor_rn})"]
    imprimir_tabela(
        "Tabela 6: melhor CART x melhor RN",
        colunas,
        {nome: [formatar(t6[m][chave]) for m in estimador] for chave, nome in LINHAS_TABELA_6.items()},
    )
    imprimir_tabela(
        "Tabela 7: teste cego, melhor CART x melhor RN",
        colunas,
        {nome: [f"{t7[m][chave]:.5f}" for m in estimador] for chave, nome in LINHAS_TABELA_7.items()},
    )
    print("\nTabela 8: matrizes de confusão no teste cego (G=verde, Y=amarelo, R=vermelho, B=preto)")
    imprimir_matriz(f"CART ({melhor_cart})", t8["cart"])
    imprimir_matriz(f"RN ({melhor_rn})", t8["rn"])
    imprimir_tabela(
        "F1 por classe: validação f1 (DPA) | teste cego",
        colunas,
        {
            cor: [f"{formatar(f1_validacao[m][cor])} | {f1_cego[m][cor]:.5f}" for m in estimador]
            for cor in CORES.values()
        },
    )
    print(f"\nTeste cego: {saida['teste_cego']}")
    print(f"Tabelas 6, 7 e 8 salvas em: {TABELAS_6_7_8.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
