"""Avalia as hiperparametrizações U, E e O da RN e salva os números das Tabelas 4 e 5.

A RN é um Pipeline com StandardScaler e MLPClassifier (item 11 de observacoes.md), para que a
padronização seja ajustada só no fold de treino. Cada hiperparametrização passa pela validação
cruzada do módulo comum (k=5, semente 42); a melhor RN sai do critério do item 10 de
observacoes.md (escolher_melhor). Os avisos de não convergência não são silenciados: aparecem
na saída, e o número de folds com aviso vai para o JSON junto com o tempo de cada avaliação.
Nunca lê o teste cego.

Uso: uv run python src/treinar_rn.py
"""

import json
import time
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from validacao_cruzada import K_FOLDS, avaliar, carregar_dataset, descrever, escolher_melhor, formatar

RAIZ = Path(__file__).resolve().parent.parent
TABELAS_4_5 = RAIZ / "resultados" / "tabelas4_5_rn.json"

RN = Pipeline([("padronizacao", StandardScaler()), ("mlp", MLPClassifier())])

# PROVISÓRIO: conjunto A, recomendado pelo protótipo do ticket #5; aguarda a escolha do usuário.
# Hiperparametrizações U, E e O (Tabela 4), com o prefixo do passo do Pipeline (item 20); as
# linhas da Tabela 4 ficam explícitas mesmo quando o valor é o padrão do scikit-learn.
# max_iter=2000 dá folga para o treino parar pelo critério de tol (convergência): com o padrão
# (200), a RN não converge nesta base (item 23). O lbfgs ignora learning_rate_init.
HIPERPARAMETRIZACOES = {
    "U": {
        "mlp__hidden_layer_sizes": (2,),
        "mlp__activation": "logistic",
        "mlp__learning_rate_init": 0.001,
        "mlp__solver": "sgd",
        "mlp__alpha": 0.0001,
        "mlp__max_iter": 2000,
    },
    "E": {
        "mlp__hidden_layer_sizes": (64,),
        "mlp__activation": "relu",
        "mlp__learning_rate_init": 0.001,
        "mlp__solver": "adam",
        "mlp__alpha": 0.0001,
        "mlp__max_iter": 2000,
    },
    "O": {
        "mlp__hidden_layer_sizes": (128, 128),
        "mlp__activation": "relu",
        "mlp__solver": "lbfgs",
        "mlp__alpha": 0.0,
        "mlp__max_iter": 2000,
    },
}

# As quatro primeiras são as linhas do enunciado; alpha e max_iter completam a hiperparametrização.
LINHAS_TABELA_4 = {
    "hidden_layer_sizes": "Topologia",
    "activation": "Função de ativação",
    "learning_rate_init": "Learning rate",
    "solver": "solver",
    "alpha": "alpha",
    "max_iter": "max_iter",
}
LINHAS_TABELA_5 = {
    "treino": "TREINO f1 (DPA)",
    "validacao": "VALIDAÇÃO f1 (DPA)",
    "diferenca": "MÉDIA DAS DIFS. (DPA)",
}


def tabela_4():
    """Linhas da Tabela 4 por hiperparametrização, sem o prefixo mlp__.

    Topologia vira lista (neurônios por camada oculta); com lbfgs, Learning rate fica None
    porque o solver não usa learning rate.
    """
    tabela = {}
    for rotulo, hiperparametrizacao in HIPERPARAMETRIZACOES.items():
        mlp = RN.named_steps["mlp"].get_params()
        mlp.update({nome.removeprefix("mlp__"): valor for nome, valor in hiperparametrizacao.items()})
        linhas = {chave: mlp[chave] for chave in LINHAS_TABELA_4}
        linhas["hidden_layer_sizes"] = list(linhas["hidden_layer_sizes"])
        if linhas["solver"] == "lbfgs":
            linhas["learning_rate_init"] = None
        tabela[rotulo] = linhas
    return tabela


def avaliar_rn(hiperparametrizacao, X, y):
    """Resultado de avaliar, mais o tempo da avaliação e o número de folds com ConvergenceWarning.

    Roda com 1 thread de BLAS: nestas RN pequenas, várias threads não aceleram e deixam o
    lbfgs muitas vezes mais lento (a O passou de 10 min sem terminar, contra ~2,5 min com 1 thread).
    Cada aviso distinto é reemitido uma vez, com a contagem.
    """
    with threadpool_limits(1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        resultado = avaliar(RN, hiperparametrizacao, X, y)
        tempo = time.perf_counter() - inicio
    distintos = {}
    for aviso in avisos:
        distintos.setdefault((aviso.category, str(aviso.message)), []).append(aviso)
    for (categoria, mensagem), repetidos in distintos.items():
        a = repetidos[0]
        warnings.showwarning(f"{mensagem} ({len(repetidos)}x)", categoria, a.filename, a.lineno)
    nao_convergiu = sum(issubclass(a.category, ConvergenceWarning) for a in avisos)
    return resultado, {"tempo_s": tempo, "folds_sem_convergir": nao_convergiu}


def celula_tabela_4(chave, valor):
    if chave == "hidden_layer_sizes":
        return "[" + " ".join(str(n) for n in valor) + "]"
    if valor is None:
        return "— (lbfgs)"
    return valor


def imprimir_tabela(titulo, linhas):
    """Tabela em texto: `linhas` mapeia o nome da linha -> células de U, E e O."""
    print(f"\n{titulo}")
    print(f"{'MODELO':<24}" + "".join(f"{rotulo:<20}" for rotulo in HIPERPARAMETRIZACOES))
    for nome, celulas in linhas.items():
        print(f"{nome:<24}" + "".join(f"{str(c):<20}" for c in celulas))


def main():
    X, y = carregar_dataset()
    resultados, execucao = {}, {}
    for rotulo, hiperparametrizacao in HIPERPARAMETRIZACOES.items():
        resultados[rotulo], execucao[rotulo] = avaliar_rn(hiperparametrizacao, X, y)
        e = execucao[rotulo]
        print(f"\n{rotulo}: {hiperparametrizacao}")
        print(f"{e['tempo_s']:.1f} s; ConvergenceWarning em {e['folds_sem_convergir']}/{K_FOLDS} folds")
        print(descrever(resultados[rotulo]))
    melhor = escolher_melhor(resultados)
    t4 = tabela_4()

    saida = {
        "tabela_4": t4,
        "tabela_5": {
            rotulo: {nome: r[nome] for nome in LINHAS_TABELA_5} for rotulo, r in resultados.items()
        },
        "melhor_rn": melhor,
        "execucao": execucao,
    }
    TABELAS_4_5.parent.mkdir(exist_ok=True)
    TABELAS_4_5.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    imprimir_tabela(
        "Tabela 4: hiperparametrizações RN",
        {nome: [celula_tabela_4(chave, h[chave]) for h in t4.values()] for chave, nome in LINHAS_TABELA_4.items()},
    )
    imprimir_tabela(
        "Tabela 5: resultados dos modelos RN",
        {nome: [formatar(r[chave]) for r in resultados.values()] for chave, nome in LINHAS_TABELA_5.items()},
    )
    print(f"\nMelhor RN: {melhor}")
    print(f"Tabelas 4 e 5 salvas em: {TABELAS_4_5.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
