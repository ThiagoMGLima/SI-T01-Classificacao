"""PROTÓTIPO (descartável) — ticket #5. F1 de validação por classe de triagem das candidatas a U.

Diagnóstico, não avaliação: junta as predições de validação dos mesmos 5 folds de `avaliar`
(cross_val_predict com o mesmo StratifiedKFold) e mede o F1 de cada classe, para ver se uma U
com F1 macro baixo perde todas as classes um pouco ou zera uma classe (item 22).

Uso: uv run python src/prototipo_varredura_rn/diagnostico_classes.py
"""

import json

from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from registro import RN, X, Y
from validacao_cruzada import K_FOLDS, SEMENTE, montar
from varredura_rn import AQUI, config

SAIDA = AQUI / "saida" / "diagnostico_classes.json"
CORES = ["verde", "amarelo", "vermelho", "preto"]

CANDIDATAS = [
    config("U", hidden_layer_sizes=[2], activation="logistic", solver="sgd", learning_rate_init=0.001),
    config("U", hidden_layer_sizes=[2], alpha=30.0),
    config("U", hidden_layer_sizes=[32, 16], alpha=30.0),
    config("U", hidden_layer_sizes=[64, 64], alpha=30.0),
    config("U", hidden_layer_sizes=[1], activation="logistic"),
    config("U", hidden_layer_sizes=[2], activation="logistic"),
    config("U", hidden_layer_sizes=[2]),
    config("E", hidden_layer_sizes=[64]),
]


def f1_por_classe(c):
    folds = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    with threadpool_limits(1):
        pred = cross_val_predict(montar(RN, c["hiper"]), X, Y, cv=folds)
    f1 = f1_score(Y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    preditas = {cor: int((pred == i).sum()) for i, cor in enumerate(CORES)}
    return {**c, "f1_validacao_por_classe": dict(zip(CORES, map(float, f1))), "n_preditas": preditas}


def main():
    resultados = Parallel(n_jobs=len(CANDIDATAS))(delayed(f1_por_classe)(c) for c in CANDIDATAS)
    SAIDA.write_text(json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in resultados:
        h = r["hiper"]
        f1 = " ".join(f"{cor}={v:.3f}" for cor, v in r["f1_validacao_por_classe"].items())
        print(f"{h['mlp__hidden_layer_sizes']} {h['mlp__activation']} {h['mlp__solver']} "
              f"alpha={h['mlp__alpha']}: F1 {f1} | preditas {r['n_preditas']}")


if __name__ == "__main__":
    main()
