"""PROTÓTIPO (descartável) — ticket #6. Gera números FICTÍCIOS para as Tabelas 2 a 8.

Imita o JSON que os scripts de treino e de teste cego provavelmente vão gravar em
`resultados/`, mas grava em `src/prototipo_relatorio/dados_ficticios/` para não se
confundir com resultados reais. Nenhum modelo é treinado aqui.

Uso: uv run python src/prototipo_relatorio/gerar_dados_ficticios.py
"""

import json
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent / "dados_ficticios"
K = 5
rng = np.random.default_rng(42)


def estatisticas(valores):
    valores = np.asarray(valores)
    return {
        "f1_por_fold": [round(float(v), 5) for v in valores],
        "media": float(valores.mean()),
        "dpa": float(valores.std(ddof=0)),
    }


def validacao_cruzada(medias_treino, medias_valid, ruido):
    """Para cada hiperparametrização, inventa F1 macro por fold de treino e de validação."""
    resultados = {}
    for nome in "UEO":
        treino = np.clip(rng.normal(medias_treino[nome], ruido, K), 0, 1)
        valid = np.clip(rng.normal(medias_valid[nome], ruido * 1.5, K), 0, 1)
        resultados[nome] = {
            "treino": estatisticas(treino),
            "validacao": estatisticas(valid),
            "diferenca": estatisticas(np.abs(treino - valid)),
        }
    return resultados


def metricas_teste_cego(matriz):
    """Precisão, recall e F1 macro e acurácia a partir da matriz de confusão (real × predito)."""
    m = np.asarray(matriz, dtype=float)
    vp = np.diag(m)
    precisao = vp / m.sum(axis=0)
    recall = vp / m.sum(axis=1)
    f1 = 2 * precisao * recall / (precisao + recall)
    return {
        "precisao_macro": float(precisao.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "acuracia": float(vp.sum() / m.sum()),
    }


def main():
    PASTA.mkdir(exist_ok=True)

    cart = {
        "k": K,
        "hiperparametrizacoes": {
            "U": {"min_samples_leaf": 300, "max_depth": 2, "criterion": "gini"},
            "E": {"min_samples_leaf": 10, "max_depth": 10, "criterion": "gini"},
            "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "entropy"},
        },
        "resultados": validacao_cruzada(
            {"U": 0.61, "E": 0.905, "O": 1.0}, {"U": 0.605, "E": 0.862, "O": 0.821}, 0.006
        ),
        "melhor": "E",
    }
    # O sobreajustado decora o treino: F1 de treino exatamente 1 em todos os folds.
    cart["resultados"]["O"]["treino"] = estatisticas([1.0] * K)
    valid_o = cart["resultados"]["O"]["validacao"]["f1_por_fold"]
    cart["resultados"]["O"]["diferenca"] = estatisticas([1.0 - v for v in valid_o])

    rn = {
        "k": K,
        "hiperparametrizacoes": {
            "U": {"hidden_layer_sizes": [2], "activation": "logistic", "learning_rate_init": 0.001, "solver": "adam"},
            "E": {"hidden_layer_sizes": [32, 16], "activation": "relu", "learning_rate_init": 0.001, "solver": "adam"},
            "O": {"hidden_layer_sizes": [256, 256, 256], "activation": "relu", "learning_rate_init": 0.01, "solver": "adam"},
        },
        "resultados": validacao_cruzada(
            {"U": 0.55, "E": 0.893, "O": 0.972}, {"U": 0.547, "E": 0.878, "O": 0.851}, 0.005
        ),
        "melhor": "E",
    }

    # Teste cego: 1300 vítimas, proporção 100:390:405:405 (antes do ruído). Linhas = real, colunas = predito.
    matriz_cart = [[78, 17, 3, 2], [14, 341, 30, 5], [2, 28, 352, 23], [0, 4, 26, 375]]
    matriz_rn = [[84, 13, 2, 1], [11, 352, 24, 3], [1, 22, 364, 18], [0, 3, 21, 381]]
    teste_cego = {
        "classes": ["verde", "amarelo", "vermelho", "preto"],
        "cart": {"hiperparametrizacao": cart["melhor"], **metricas_teste_cego(matriz_cart), "matriz_confusao": matriz_cart},
        "rn": {"hiperparametrizacao": rn["melhor"], **metricas_teste_cego(matriz_rn), "matriz_confusao": matriz_rn},
    }

    for nome, conteudo in [
        ("cart_validacao_cruzada.json", cart),
        ("rn_validacao_cruzada.json", rn),
        ("teste_cego.json", teste_cego),
    ]:
        (PASTA / nome).write_text(json.dumps(conteudo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("gravado", PASTA / nome)


if __name__ == "__main__":
    main()
