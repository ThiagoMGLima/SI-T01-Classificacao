"""Validação cruzada e seleção do melhor CART / melhor RN.

Módulo comum aos scripts de treino. Avalia uma hiperparametrização com StratifiedKFold
(k=5, embaralhado, semente 42) no dataset de treino/validação, usando só as 10 características
de entrada, e escolhe entre U, E e O pelo critério do item 10 de observacoes.md.
Nunca lê o teste cego.

Uso nos scripts de src/ (rodados com `uv run python src/<script>.py`):
    from validacao_cruzada import avaliar, carregar_dataset, escolher_melhor, formatar

Verificação com o exemplo do enunciado: uv run python src/validacao_cruzada.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "dados" / "treino_validacao_10000v.csv"

CARACTERISTICAS = ["idade", "fc", "fr", "pas", "spo2", "temp", "pr", "sg", "fx", "queim"]
CLASSE = "tri"

K_FOLDS = 5
SEMENTE = 42
MARGEM_EMPATE = 0.005  # empate técnico no F1 macro médio de validação (item 10)

# F1 macro: média simples do F1 das quatro classes de triagem. Uma classe que o modelo
# nunca prediz entra com F1 0 (o mesmo valor do padrão do scikit-learn, sem o aviso).
F1_MACRO = make_scorer(f1_score, average="macro", zero_division=0)

# Exemplo do enunciado (Tabela 3): F1 macro por fold de um CART com k=3.
EXEMPLO_TREINO = [0.86710, 0.88764, 0.89162]
EXEMPLO_VALIDACAO = [0.85837, 0.88636, 0.87903]
EXEMPLO_ESPERADO = {
    "treino": "0.88212 (0.01074)",
    "validacao": "0.87459 (0.01185)",
    "diferenca": "0.00753 (0.00469)",
}


def carregar_dataset():
    """Dataset de treino/validação: X com as 10 características de entrada e y = classe de triagem."""
    df = pd.read_csv(DATASET)
    return df[CARACTERISTICAS], df[CLASSE]


def montar(estimador, hiperparametrizacao):
    """Cópia não ajustada do estimador com a hiperparametrização aplicada.

    Todo `random_state` deixado em None, inclusive dentro de um Pipeline (ex.: `mlp__random_state`),
    recebe a semente 42. O retreino deve montar o modelo por aqui para reproduzir o da validação.
    """
    modelo = clone(estimador).set_params(**hiperparametrizacao)
    sem_semente = {
        nome: SEMENTE
        for nome, valor in modelo.get_params().items()
        if nome.split("__")[-1] == "random_state" and valor is None
    }
    return modelo.set_params(**sem_semente)


def _estatisticas(valores):
    return {
        "por_fold": [float(v) for v in valores],
        "media": float(np.mean(valores)),
        "dpa": float(np.std(valores, ddof=0)),
    }


def resumir_folds(f1_treino, f1_validacao):
    """Por fold, média e DPA (ddof=0) do F1 macro de treino, de validação e da diferença treino-validação.

    Devolve {"treino", "validacao", "diferenca"}, cada um com "por_fold", "media" e "dpa".
    A diferença treino-validação é o valor absoluto, fold a fold.
    """
    treino = np.asarray(f1_treino, dtype=float)
    validacao = np.asarray(f1_validacao, dtype=float)
    return {
        "treino": _estatisticas(treino),
        "validacao": _estatisticas(validacao),
        "diferenca": _estatisticas(np.abs(treino - validacao)),
    }


def avaliar(estimador, hiperparametrizacao, X=None, y=None):
    """Avalia uma hiperparametrização com validação cruzada estratificada (k=5, semente 42).

    Sem X e y, carrega o dataset de treino/validação (passe-os para não reler o CSV a cada
    chamada). Os folds são sempre os mesmos, para qualquer estimador e hiperparametrização.
    Devolve um dicionário serializável em JSON: {"hiperparametrizacao", "treino", "validacao",
    "diferenca"}, os três últimos como em resumir_folds.
    """
    if X is None:
        X, y = carregar_dataset()
    folds = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    cv = cross_validate(
        montar(estimador, hiperparametrizacao),
        X,
        y,
        cv=folds,
        scoring=F1_MACRO,
        return_train_score=True,
    )
    return {
        "hiperparametrizacao": dict(hiperparametrizacao),
        **resumir_folds(cv["train_score"], cv["test_score"]),
    }


def escolher_melhor(resultados):
    """Rótulo do melhor CART / melhor RN entre U, E e O, pelo critério do item 10 de observacoes.md.

    `resultados` mapeia rótulo -> resultado de avaliar, ex.: {"U": ..., "E": ..., "O": ...}.
    Empate técnico entre quem tiver F1 macro médio de validação a até 0,005 do maior; desempate,
    nesta ordem, por menor DPA de validação, menor média da diferença treino-validação e maior
    F1 macro médio de validação. Se ainda houver empate, vence o primeiro na ordem do dicionário.
    """
    maior = max(r["validacao"]["media"] for r in resultados.values())
    empatados = [
        rotulo for rotulo, r in resultados.items() if r["validacao"]["media"] >= maior - MARGEM_EMPATE
    ]
    return min(
        empatados,
        key=lambda rotulo: (
            resultados[rotulo]["validacao"]["dpa"],
            resultados[rotulo]["diferenca"]["media"],
            -resultados[rotulo]["validacao"]["media"],
        ),
    )


def formatar(estatistica):
    """Célula das Tabelas 3, 5 e 6: média (DPA) com 5 casas, ex.: "0.88212 (0.01074)"."""
    return f"{estatistica['media']:.5f} ({estatistica['dpa']:.5f})"


def descrever(resultado):
    """Texto no formato do exemplo do enunciado: F1 por fold, média e DPA."""
    linhas = []
    for rotulo, nome in [
        ("Treino:    F1 por fold:", "treino"),
        ("Validação: F1 por fold:", "validacao"),
        ("Diferenças abs........:", "diferenca"),
    ]:
        e = resultado[nome]
        por_fold = ", ".join(f"{v:.5f}" for v in e["por_fold"])
        linhas.append(f"{rotulo} [{por_fold}] média={e['media']:.5f} dpa={e['dpa']:.5f}")
    return "\n".join(linhas)


def main():
    """Confere o cálculo de média, DPA e diferença treino-validação com o exemplo do enunciado."""
    resumo = resumir_folds(EXEMPLO_TREINO, EXEMPLO_VALIDACAO)
    print(descrever(resumo))
    for nome, esperado in EXEMPLO_ESPERADO.items():
        obtido = formatar(resumo[nome])
        if obtido != esperado:
            raise SystemExit(f"{nome}: obtido {obtido}, esperado {esperado}")
        print(f"{nome:<10} {obtido}  (enunciado: {esperado})")
    print("Exemplo do enunciado reproduzido.")


if __name__ == "__main__":
    main()
