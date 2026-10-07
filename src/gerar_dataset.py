"""Gera o dataset de treino/validação (10.000 vítimas) e salva os números da Tabela 1.

Uso: uv run python src/gerar_dataset.py
"""

import contextlib
import importlib.util
import json
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # o gerador chama plt.show(); sem janela
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
GERADOR = RAIZ / "vendor" / "victsim3" / "gerar_dados_vitimas.py"
DATASET = RAIZ / "dados" / "treino_validacao_10000v.csv"
TABELA_1 = RAIZ / "resultados" / "tabela1_dataset.json"

# Configuração do main do gerador (a que gerou o teste cego), escalada para 10.000 vítimas:
# proporção 100:390:405:405 entre as classes de triagem antes do ruído.
DISTRIB_TRI = {0: 769, 1: 3000, 2: 3115, 3: 3116}
MEDIA_IDADE = 40
DESVIO_IDADE = 25
NIVEL_RUIDO = 0.05
SEMENTE = 42

CORES = {0: "verde", 1: "amarelo", 2: "vermelho", 3: "preto"}


def carregar_gerador():
    """Importa o gerador do VictSim3 sem modificá-lo.

    No import, ele cria ./datasets/vict/1300v/ no diretório atual; importamos dentro
    de um diretório temporário para não criar essa pasta no repositório.
    """
    spec = importlib.util.spec_from_file_location("gerar_dados_vitimas", GERADOR)
    gerador = importlib.util.module_from_spec(spec)
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        spec.loader.exec_module(gerador)
    return gerador


def gerar_dataset():
    gerador = carregar_gerador()
    # O gerador sempre salva no global OUTPUT_CSV; apontamos para dados/
    gerador.OUTPUT_CSV = DATASET
    DATASET.parent.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-interactive.*")  # plt.show() no Agg
        df = gerador.gerar_dataset_vitimas(
            distrib_tri=DISTRIB_TRI,
            media_idade=MEDIA_IDADE,
            desvio_idade=DESVIO_IDADE,
            nivel_ruido=NIVEL_RUIDO,
            seed=SEMENTE,
        )
    plt.close("all")
    return df


def numeros_tabela_1(df):
    """Números da Tabela 1, medidos no dataset gerado (após o ruído na classe de triagem)."""
    contagem = df["tri"].value_counts()
    return {
        "vitimas_por_classe": {cor: int(contagem.get(tri, 0)) for tri, cor in CORES.items()},
        "idade_media": float(df["idade"].mean()),
        "dpa_idade": float(df["idade"].std(ddof=0)),
        "ruido": NIVEL_RUIDO,
    }


def main():
    df = gerar_dataset()
    resultado = {
        "parametros_gerador": {
            "n_vitimas": sum(DISTRIB_TRI.values()),
            "distrib_tri_antes_do_ruido": {CORES[tri]: n for tri, n in DISTRIB_TRI.items()},
            "media_idade": MEDIA_IDADE,
            "desvio_idade": DESVIO_IDADE,
            "nivel_ruido": NIVEL_RUIDO,
            "semente": SEMENTE,
        },
        "tabela_1": numeros_tabela_1(df),
    }
    TABELA_1.parent.mkdir(exist_ok=True)
    TABELA_1.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nTabela 1 salva em: {TABELA_1.relative_to(RAIZ)}")
    print(json.dumps(resultado["tabela_1"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
