"""PROTÓTIPO (descartável) — ticket #5. Mede o tempo da RN padrão com e sem limite de threads."""

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

from validacao_cruzada import avaliar, carregar_dataset, formatar

X, y = carregar_dataset()
rn = Pipeline([("padronizacao", StandardScaler()), ("mlp", MLPClassifier())])
for threads in [None, 1]:
    with threadpool_limits(threads), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        t = time.perf_counter()
        r = avaliar(rn, {}, X, y)
        dt = time.perf_counter() - t
    n = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    print(threads, f"{dt:.1f}s", formatar(r["treino"]), formatar(r["validacao"]), formatar(r["diferenca"]), "avisos", n)
