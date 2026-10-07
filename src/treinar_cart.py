"""Avalia as hiperparametrizações U, E e O do CART e salva os números das Tabelas 2 e 3.

Cada hiperparametrização passa pela validação cruzada do módulo comum (k=5, semente 42);
o melhor CART sai do critério do item 10 de observacoes.md (escolher_melhor).
Nunca lê o teste cego.

Uso: uv run python src/treinar_cart.py
"""

import json
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from validacao_cruzada import avaliar, carregar_dataset, descrever, escolher_melhor, formatar

RAIZ = Path(__file__).resolve().parent.parent
TABELAS_2_3 = RAIZ / "resultados" / "tabelas2_3_cart.json"

# Hiperparametrizações U, E e O (Tabela 2), escolhidas com o usuário no ticket #4 (conjunto A da
# varredura do branch prototipo/varredura-cart). As três linhas obrigatórias da Tabela 2 ficam
# explícitas mesmo quando o valor é o padrão do scikit-learn (min_samples_leaf=1, max_depth=None).
# max_depth=None (sem limite) vai como null no JSON; o relatório o exibe como "None".
HIPERPARAMETRIZACOES = {
    "U": {"min_samples_leaf": 1, "max_depth": 2, "criterion": "entropy"},
    "E": {"min_samples_leaf": 8, "max_depth": 8, "criterion": "entropy"},
    "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "entropy"},
}

LINHAS_TABELA_2 = {
    "min_samples_leaf": "Min_samples_leaf",
    "max_depth": "Max_depth",
    "criterion": "Criterion (gini, ent)",
}
LINHAS_TABELA_3 = {
    "treino": "TREINO f1 (DPA)",
    "validacao": "VALIDAÇÃO f1 (DPA)",
    "diferenca": "MÉDIA DAS DIFS. (DPA)",
}


def imprimir_tabela(titulo, linhas):
    """Tabela em texto: `linhas` mapeia o nome da linha -> células de U, E e O."""
    print(f"\n{titulo}")
    print(f"{'MODELO':<24}" + "".join(f"{rotulo:<20}" for rotulo in HIPERPARAMETRIZACOES))
    for nome, celulas in linhas.items():
        print(f"{nome:<24}" + "".join(f"{str(c):<20}" for c in celulas))


def main():
    X, y = carregar_dataset()
    resultados = {}
    for rotulo, hiperparametrizacao in HIPERPARAMETRIZACOES.items():
        resultados[rotulo] = avaliar(DecisionTreeClassifier(), hiperparametrizacao, X, y)
        print(f"\n{rotulo}: {hiperparametrizacao}")
        print(descrever(resultados[rotulo]))
    melhor = escolher_melhor(resultados)

    saida = {
        "tabela_2": HIPERPARAMETRIZACOES,
        "tabela_3": {
            rotulo: {nome: r[nome] for nome in LINHAS_TABELA_3} for rotulo, r in resultados.items()
        },
        "melhor_cart": melhor,
    }
    TABELAS_2_3.parent.mkdir(exist_ok=True)
    TABELAS_2_3.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    imprimir_tabela(
        "Tabela 2: hiperparametrizações CART",
        {nome: [h[chave] for h in HIPERPARAMETRIZACOES.values()] for chave, nome in LINHAS_TABELA_2.items()},
    )
    imprimir_tabela(
        "Tabela 3: resultados dos modelos CART",
        {nome: [formatar(r[chave]) for r in resultados.values()] for chave, nome in LINHAS_TABELA_3.items()},
    )
    print(f"\nMelhor CART: {melhor}")
    print(f"Tabelas 2 e 3 salvas em: {TABELAS_2_3.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
