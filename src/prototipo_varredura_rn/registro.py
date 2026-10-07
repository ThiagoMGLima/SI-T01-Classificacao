"""PROTÓTIPO (descartável) — ticket #5 "Escolher as hiperparametrizações U/E/O da RN".

Avalia uma hiperparametrização da RN com `avaliar` (src/validacao_cruzada.py) e registra,
além do F1 macro, o tempo da avaliação, os avisos de não convergência e as épocas usadas
em cada fold. Fica num módulo próprio para que os processos paralelos da varredura o
importem por nome.

A RN da varredura troca MLPClassifier por MLPRegistrada, uma subclasse que só anota
n_iter_ e loss_ ao fim de cada fit; o treino e o F1 são os mesmos do MLPClassifier.
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from validacao_cruzada import avaliar, carregar_dataset

# Épocas (n_iter_) e perda final de treino de cada fit, na ordem dos folds.
_FITS = []


class MLPRegistrada(MLPClassifier):
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        _FITS.append({"n_iter": int(self.n_iter_), "loss": float(self.loss_)})
        return self


RN = Pipeline([("padronizacao", StandardScaler()), ("mlp", MLPRegistrada())])
X, Y = carregar_dataset()


def n_parametros(topologia, n_entradas=10, n_saidas=4):
    """Pesos + vieses da RN: medida de complexidade da topologia."""
    camadas = [n_entradas, *topologia, n_saidas]
    return sum((a + 1) * b for a, b in zip(camadas, camadas[1:]))


def avaliar_config(config):
    """Avalia {"id", "eixo", "hiper"} com 1 thread de BLAS; devolve o resultado de avaliar + registros."""
    _FITS.clear()
    with threadpool_limits(1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        resultado = avaliar(RN, config["hiper"], X, Y)
        tempo = time.perf_counter() - inicio
    nao_convergiu = [a for a in avisos if issubclass(a.category, ConvergenceWarning)]
    outros = sorted({f"{a.category.__name__}: {a.message}" for a in avisos if a not in nao_convergiu})
    topologia = config["hiper"].get("mlp__hidden_layer_sizes", [100])
    return {
        **config,
        "resultado": resultado,
        "tempo_s": tempo,
        "avisos_convergencia": len(nao_convergiu),
        "mensagem_convergencia": str(nao_convergiu[0].message) if nao_convergiu else None,
        "outros_avisos": outros,
        "epocas_por_fold": [f["n_iter"] for f in _FITS],
        "perda_final_por_fold": [f["loss"] for f in _FITS],
        "n_parametros": n_parametros(topologia),
    }
