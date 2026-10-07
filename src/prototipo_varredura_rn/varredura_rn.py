"""PROTÓTIPO (descartável) — ticket #5 "Escolher as hiperparametrizações U/E/O da RN".

Varredura exploratória do F1 macro de treino × validação da RN (Pipeline StandardScaler +
MLPClassifier) conforme a complexidade: topologia, função de ativação, alpha, solver e
learning rate, e max_iter. Cada hiperparametrização é avaliada com `avaliar`
(src/validacao_cruzada.py: k=5 estratificado, semente 42), em paralelo, 1 thread por processo.

Os avisos de não convergência NÃO são silenciados: cada avaliação registra em quantos folds
apareceu ConvergenceWarning e quantas épocas cada fold usou (ver registro.py).

Uso:
    uv run python src/prototipo_varredura_rn/varredura_rn.py grossa                  # varredura grossa
    uv run python src/prototipo_varredura_rn/varredura_rn.py subajuste sobreajuste   # busca da U e da O
    uv run python src/prototipo_varredura_rn/varredura_rn.py candidatos              # conjuntos U/E/O

Resultados acumulam em saida/varredura_rn.jsonl (uma avaliação por linha, chave "id");
avaliações já feitas são puladas. Gráficos e tabela: graficos_rn.py.
"""

import json
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

from registro import avaliar_config, n_parametros

AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "saida" / "varredura_rn.jsonl"
N_PROCESSOS = 12  # de 16 núcleos; deixa folga para o outro ticket

# Ponto de partida de toda a varredura: o padrão do MLPClassifier, com max_iter folgado
# para que a parada seja pelo critério de tol (convergência) e não pelo limite de épocas.
BASE = {
    "mlp__hidden_layer_sizes": [100],
    "mlp__activation": "relu",
    "mlp__solver": "adam",
    "mlp__learning_rate_init": 0.001,
    "mlp__alpha": 0.0001,
    "mlp__max_iter": 2000,
}


def config(eixo, **hiper):
    h = {**BASE, **{f"mlp__{nome}": valor for nome, valor in hiper.items()}}
    return {"id": eixo + " " + json.dumps(h, sort_keys=True), "eixo": eixo, "hiper": h}


def etapa_grossa():
    c = []
    # Topologia: largura (1 camada) e profundidade.
    for n in [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256]:
        c.append(config("topologia", hidden_layer_sizes=[n]))
    for t in [[8, 4], [16, 8], [32, 16], [32, 32], [32, 32, 32], [64, 64], [64, 64, 64], [128, 128], [256, 256]]:
        c.append(config("topologia", hidden_layer_sizes=t))
    # Função de ativação, numa topologia média.
    for a in ["identity", "logistic", "tanh", "relu"]:
        c.append(config("ativacao", hidden_layer_sizes=[32, 16], activation=a))
    # Regularização L2 (alpha), numa topologia com folga para sobreajustar.
    for alpha in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        c.append(config("alpha", hidden_layer_sizes=[64, 64], alpha=alpha))
    # Solver e learning rate (lbfgs ignora learning_rate_init).
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        c.append(config("solver_lr", hidden_layer_sizes=[32, 16], solver="adam", learning_rate_init=lr))
    for lr in [0.001, 0.01, 0.1]:
        c.append(config("solver_lr", hidden_layer_sizes=[32, 16], solver="sgd", learning_rate_init=lr))
    c.append(config("solver_lr", hidden_layer_sizes=[32, 16], solver="lbfgs"))
    # Épocas: a RN padrão ([100], relu, adam) com max_iter crescente (200 = item 23).
    for m in [5, 10, 20, 50, 100, 200, 500, 2000]:
        c.append(config("max_iter", max_iter=m))
    return c


def etapa_subajuste():
    """U: F1 baixo no treino e na validação, convergindo e sem colapsar numa classe só."""
    c = []
    for n in [1, 2]:
        for a in ["identity", "logistic", "tanh"]:
            c.append(config("subajuste", hidden_layer_sizes=[n], activation=a))
    for alpha in [3.0, 10.0, 30.0]:
        c.append(config("subajuste", hidden_layer_sizes=[2], alpha=alpha))
        c.append(config("subajuste", hidden_layer_sizes=[32, 16], alpha=alpha))
        c.append(config("subajuste", hidden_layer_sizes=[100], alpha=alpha))
    for alpha in [15.0, 20.0]:
        c.append(config("subajuste", hidden_layer_sizes=[64, 64], alpha=alpha))
    # learning rate muito baixo: o critério de tol pode parar o treino cedo, sem aviso
    c.append(config("subajuste", hidden_layer_sizes=[32, 16], solver="sgd", learning_rate_init=0.0001))
    c.append(config("subajuste", hidden_layer_sizes=[2], activation="logistic", solver="sgd",
                    learning_rate_init=0.001))
    return c


def etapa_sobreajuste():
    """O: F1 de treino alto e diferença treino-validação grande. Alavancas: largura e profundidade,
    alpha=0, treinar por mais épocas (n_iter_no_change maior), lbfgs, tanh."""
    c = []
    for t in [[128, 128], [256, 256], [128, 128, 128], [256, 256, 256]]:
        c.append(config("sobreajuste", hidden_layer_sizes=t, alpha=0.0))
    c.append(config("sobreajuste", hidden_layer_sizes=[128, 128], alpha=0.0, n_iter_no_change=50))
    c.append(config("sobreajuste", hidden_layer_sizes=[128, 128, 128], alpha=0.0, activation="tanh"))
    c.append(config("sobreajuste", hidden_layer_sizes=[128, 128], alpha=0.0, learning_rate_init=0.01))
    for t in [[64, 64], [128, 128], [256, 256]]:
        c.append(config("sobreajuste", hidden_layer_sizes=t, alpha=0.0, solver="lbfgs"))
    return c


def _hiper(**hiper):
    return config("", **hiper)["hiper"]


# Conjuntos candidatos a U/E/O (todas as hiperparametrizações já avaliadas nas etapas acima).
CONJUNTOS = {
    "A": {  # capacidade e otimização crescem juntas (recomendado)
        "U": _hiper(hidden_layer_sizes=[2], activation="logistic", solver="sgd"),
        "E": _hiper(hidden_layer_sizes=[64]),
        "O": _hiper(hidden_layer_sizes=[128, 128], alpha=0.0, solver="lbfgs"),
    },
    "B": {  # só adam: a linha Learning rate vale para as três
        "U": _hiper(hidden_layer_sizes=[1], activation="logistic"),
        "E": _hiper(hidden_layer_sizes=[32, 16]),
        "O": _hiper(hidden_layer_sizes=[128, 128, 128], activation="tanh", alpha=0.0),
    },
    "C": {  # mesma topologia [64 64]: só a regularização (e o solver da O) muda
        "U": _hiper(hidden_layer_sizes=[64, 64], alpha=30.0),
        "E": _hiper(hidden_layer_sizes=[64, 64], alpha=0.1),
        "O": _hiper(hidden_layer_sizes=[64, 64], alpha=0.0, solver="lbfgs"),
    },
}


def etapa_candidatos():
    return []


ETAPAS = {
    "grossa": etapa_grossa,
    "subajuste": etapa_subajuste,
    "sobreajuste": etapa_sobreajuste,
    "candidatos": etapa_candidatos,
}


def ler_cache():
    if not CACHE.exists():
        return {}
    linhas = CACHE.read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in map(json.loads, linhas)}


def resumo(r):
    res = r["resultado"]
    h = r["hiper"]
    nome = (
        f"{h['mlp__hidden_layer_sizes']} {h['mlp__activation']} {h['mlp__solver']} "
        f"lr={h['mlp__learning_rate_init']} alpha={h['mlp__alpha']} max_iter={h['mlp__max_iter']}"
    )
    extras = {k: v for k, v in h.items() if k not in BASE}
    return (
        f"[{r['eixo']}] {nome} {extras or ''}\n"
        f"    treino {res['treino']['media']:.5f} ({res['treino']['dpa']:.5f})  "
        f"validação {res['validacao']['media']:.5f} ({res['validacao']['dpa']:.5f})  "
        f"dif {res['diferenca']['media']:.5f} ({res['diferenca']['dpa']:.5f})  "
        f"{r['tempo_s']:.1f}s  avisos {r['avisos_convergencia']}/5  épocas {r['epocas_por_fold']}"
    )


def main():
    etapa = " + ".join(sys.argv[1:])
    configs = [c for nome in sys.argv[1:] for c in ETAPAS[nome]()]
    feitos = ler_cache()
    pendentes = [c for c in configs if c["id"] not in feitos]
    # As maiores RN primeiro, para não ficarem sozinhas no fim (a ordem não muda os números).
    pendentes.sort(key=lambda c: -n_parametros(c["hiper"]["mlp__hidden_layer_sizes"]))
    print(f"Etapa {etapa}: {len(configs)} hiperparametrizações, {len(pendentes)} a avaliar", flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    inicio = time.perf_counter()
    with CACHE.open("a", encoding="utf-8") as saida:
        tarefas = Parallel(n_jobs=N_PROCESSOS, return_as="generator_unordered")(
            delayed(avaliar_config)(c) for c in pendentes
        )
        for r in tarefas:
            saida.write(json.dumps(r, ensure_ascii=False) + "\n")
            saida.flush()
            print(resumo(r), flush=True)
    print(f"Etapa {etapa} concluída em {time.perf_counter() - inicio:.0f}s", flush=True)


if __name__ == "__main__":
    main()
