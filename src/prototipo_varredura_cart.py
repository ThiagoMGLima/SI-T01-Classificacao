"""PROTÓTIPO DESCARTÁVEL (ticket #4): varredura exploratória do CART para escolher U, E e O.

Mede o F1 macro médio de treino e de validação (com DPA) e a diferença treino-validação
conforme a complexidade da árvore, usando o módulo de validação cruzada:
- max_depth de 1 a 30 e sem limite, com min_samples_leaf=1, para gini e entropy;
- min_samples_leaf de 1 a 1000 em algumas profundidades, para gini e entropy;
- refinamento perto do pico da validação (max_depth 7 a 16 e sem limite, min_samples_leaf 5 a 15).
Depois avalia os conjuntos candidatos de U/E/O e aplica escolher_melhor em cada um.

Vive só no branch prototipo/varredura-cart, fora da main. Nunca lê o teste cego.

Uso: uv run python src/prototipo_varredura_cart.py
Saída em resultados/prototipo_varredura_cart/: varredura.csv, tabela_varredura.md,
candidatos.json, f1_vs_max_depth.png e f1_vs_min_samples_leaf.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from validacao_cruzada import avaliar, carregar_dataset, escolher_melhor, formatar

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "resultados" / "prototipo_varredura_cart"

CRITERIOS = ["gini", "entropy"]
PROFUNDIDADES = list(range(1, 31)) + [None]  # a árvore completa tem profundidade 31
FOLHAS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
PROFUNDIDADES_FOLHAS = [4, 6, 8, 10, 12, None]  # profundidades da varredura de min_samples_leaf
PROFUNDIDADES_GRAFICO_FOLHAS = [4, 8, 12, None]
# Refinamento perto do pico da validação
REFINO_PROFUNDIDADES = [*range(7, 17), None]
REFINO_FOLHAS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# F1 macro médio de validação de quem sempre prediz a classe mais frequente (item 22).
F1_CLASSE_MAIS_FREQUENTE = 0.12061

# Conjuntos candidatos de U/E/O para apresentar ao usuário (a escolha final é dele).
CANDIDATOS = {
    # entropy nas três; a complexidade cresce com max_depth e cai com min_samples_leaf
    "A": {
        "U": {"min_samples_leaf": 1, "max_depth": 2, "criterion": "entropy"},
        "E": {"min_samples_leaf": 8, "max_depth": 8, "criterion": "entropy"},
        "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "entropy"},
    },
    # gini (o padrão) nas três; O é exatamente a árvore padrão do scikit-learn (item 23)
    "B": {
        "U": {"min_samples_leaf": 1, "max_depth": 2, "criterion": "gini"},
        "E": {"min_samples_leaf": 7, "max_depth": 12, "criterion": "gini"},
        "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "gini"},
    },
    # só min_samples_leaf varia (entropy, sem limite de profundidade)
    "C": {
        "U": {"min_samples_leaf": 750, "max_depth": None, "criterion": "entropy"},
        "E": {"min_samples_leaf": 10, "max_depth": None, "criterion": "entropy"},
        "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "entropy"},
    },
}
RECOMENDADO = "A"

# Cores (paleta de referência da skill dataviz, validada para daltonismo no fundo claro).
COR_TREINO = "#2a78d6"
COR_VALIDACAO = "#eb6834"
COR_DIFERENCA = "#52514e"
COR_FUNDO = "#fcfcfb"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_2 = "#52514e"
COR_GRADE = "#e4e3df"

POSICAO_SEM_LIMITE = 32  # posição no eixo x de max_depth=None


def rotulo_profundidade(d):
    return "sem limite" if d is None or pd.isna(d) else str(int(d))


def varrer(X, y):
    """Avalia cada combinação (criterion, max_depth, min_samples_leaf) uma única vez."""
    combinacoes = []
    for criterio in CRITERIOS:
        combinacoes += [(criterio, d, 1) for d in PROFUNDIDADES]
        combinacoes += [(criterio, d, f) for d in PROFUNDIDADES_FOLHAS for f in FOLHAS]
        combinacoes += [(criterio, d, f) for d in REFINO_PROFUNDIDADES for f in REFINO_FOLHAS]
    linhas = []
    for criterio, profundidade, folha in dict.fromkeys(combinacoes):
        hiper = {"criterion": criterio, "max_depth": profundidade, "min_samples_leaf": folha}
        r = avaliar(DecisionTreeClassifier(), hiper, X, y)
        linhas.append(
            {
                **hiper,
                **{f"{nome}_{est}": r[nome][est] for nome in ["treino", "validacao", "diferenca"] for est in ["media", "dpa"]},
            }
        )
        print(f"{criterio:<8} max_depth={rotulo_profundidade(profundidade):<10} min_samples_leaf={folha:<5}"
              f" treino {formatar(r['treino'])}  validação {formatar(r['validacao'])}  dif {formatar(r['diferenca'])}")
    return pd.DataFrame(linhas)


def celula(linha, nome):
    return f"{linha[f'{nome}_media']:.5f} ({linha[f'{nome}_dpa']:.5f})"


def estilo_eixo(ax):
    ax.set_facecolor(COR_FUNDO)
    ax.grid(True, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(COR_GRADE)
    ax.tick_params(colors=COR_TEXTO_2, labelsize=9)


def curva(ax, x, df, nome, cor, rotulo):
    media, dpa = df[f"{nome}_media"].to_numpy(), df[f"{nome}_dpa"].to_numpy()
    ax.fill_between(x, media - dpa, media + dpa, color=cor, alpha=0.18, linewidth=0)
    ax.plot(x, media, color=cor, linewidth=2, marker="o", markersize=4, label=rotulo)


def grafico_profundidade(df):
    fig, eixos = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row",
                              gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=COR_FUNDO)
    for coluna, criterio in enumerate(CRITERIOS):
        sel = df[(df["criterion"] == criterio) & (df["min_samples_leaf"] == 1)].copy()
        sel["x"] = sel["max_depth"].map(lambda d: POSICAO_SEM_LIMITE if pd.isna(d) else d)
        sel = sel.sort_values("x")
        x = sel["x"].to_numpy()

        ax = eixos[0, coluna]
        estilo_eixo(ax)
        curva(ax, x, sel, "treino", COR_TREINO, "treino")
        curva(ax, x, sel, "validacao", COR_VALIDACAO, "validação")
        ax.axhline(F1_CLASSE_MAIS_FREQUENTE, color=COR_TEXTO_2, linewidth=1, linestyle=":")
        ax.text(30.5, F1_CLASSE_MAIS_FREQUENTE + 0.012, "sempre a classe mais frequente (0.12061)",
                ha="right", fontsize=8, color=COR_TEXTO_2)
        pico = sel.loc[sel["validacao_media"].idxmax()]
        ax.plot(pico["x"], pico["validacao_media"], marker="o", markersize=9, color=COR_VALIDACAO,
                markeredgecolor=COR_FUNDO, markeredgewidth=2, zorder=5)
        ax.annotate(f"pico da validação\nmax_depth={rotulo_profundidade(pico['max_depth'])}: {pico['validacao_media']:.5f}",
                    (pico["x"], pico["validacao_media"]), xytext=(pico["x"] + 2, 0.72), fontsize=9,
                    color=COR_TEXTO, arrowprops={"arrowstyle": "-", "color": COR_TEXTO_2, "linewidth": 0.8})
        perto_do_fim = sel[sel["max_depth"] == 24].iloc[0]  # rótulos diretos das curvas
        ax.text(24, perto_do_fim["treino_media"] + 0.03, "treino", color=COR_TEXTO, fontsize=9, ha="center")
        ax.text(24, perto_do_fim["validacao_media"] - 0.06, "validação", color=COR_TEXTO, fontsize=9, ha="center")
        ax.set_title(f"criterion = {criterio}", fontsize=11, color=COR_TEXTO, loc="left")
        ax.set_ylim(0.0, 1.03)
        if coluna == 0:
            ax.set_ylabel("F1 macro médio (faixa: ± DPA)", color=COR_TEXTO_2)
            ax.legend(loc="lower right", frameon=False, fontsize=9)

        ax = eixos[1, coluna]
        estilo_eixo(ax)
        curva(ax, x, sel, "diferenca", COR_DIFERENCA, "diferença treino-validação")
        ax.set_xlabel("max_depth (min_samples_leaf = 1)", color=COR_TEXTO_2)
        if coluna == 0:
            ax.set_ylabel("média das difs. (± DPA)", color=COR_TEXTO_2)

    ticks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, POSICAO_SEM_LIMITE]
    for ax in eixos[1]:
        ax.set_xticks(ticks, [str(t) for t in ticks[:-1]] + ["sem\nlimite"])
    fig.suptitle("CART: F1 macro de treino × validação conforme max_depth (k=5, StratifiedKFold)",
                 fontsize=12, color=COR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SAIDA / "f1_vs_max_depth.png", dpi=130, facecolor=COR_FUNDO)
    plt.close(fig)


def grafico_folhas(df):
    n = len(PROFUNDIDADES_GRAFICO_FOLHAS)
    fig, eixos = plt.subplots(2, n, figsize=(4 * n, 7.5), sharex=True, sharey=True, facecolor=COR_FUNDO)
    for linha, criterio in enumerate(CRITERIOS):
        for coluna, profundidade in enumerate(PROFUNDIDADES_GRAFICO_FOLHAS):
            prof = df["max_depth"].isna() if profundidade is None else df["max_depth"] == profundidade
            sel = df[(df["criterion"] == criterio) & prof]  # inclui os pontos do refinamento
            sel = sel.sort_values("min_samples_leaf")
            x = sel["min_samples_leaf"].to_numpy()
            ax = eixos[linha, coluna]
            estilo_eixo(ax)
            curva(ax, x, sel, "treino", COR_TREINO, "treino")
            curva(ax, x, sel, "validacao", COR_VALIDACAO, "validação")
            pico = sel.loc[sel["validacao_media"].idxmax()]
            ax.plot(pico["min_samples_leaf"], pico["validacao_media"], marker="o", markersize=9,
                    color=COR_VALIDACAO, markeredgecolor=COR_FUNDO, markeredgewidth=2, zorder=5)
            ax.text(1, 0.62, f"pico: leaf={int(pico['min_samples_leaf'])}\n{pico['validacao_media']:.5f}",
                    fontsize=8.5, color=COR_TEXTO)
            ax.set_xscale("log")
            ax.set_title(f"{criterio}, max_depth = {rotulo_profundidade(profundidade)}", fontsize=10,
                         color=COR_TEXTO, loc="left")
            ax.set_ylim(0.55, 1.02)
            if linha == 1:
                ax.set_xlabel("min_samples_leaf (escala log)", color=COR_TEXTO_2)
            if coluna == 0:
                ax.set_ylabel("F1 macro médio (± DPA)", color=COR_TEXTO_2)
    fig.legend(*eixos[0, 0].get_legend_handles_labels(), loc="upper right", ncols=2, frameon=False, fontsize=9)
    for ax in eixos[1]:
        ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"])
    fig.suptitle("CART: F1 macro de treino × validação conforme min_samples_leaf",
                 fontsize=12, color=COR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SAIDA / "f1_vs_min_samples_leaf.png", dpi=130, facecolor=COR_FUNDO)
    plt.close(fig)


def avaliar_candidatos(X, y):
    resultado = {}
    for nome, conjunto in CANDIDATOS.items():
        avaliados = {rotulo: avaliar(DecisionTreeClassifier(), hiper, X, y) for rotulo, hiper in conjunto.items()}
        resultado[nome] = {"resultados": avaliados, "melhor_cart": escolher_melhor(avaliados)}
    return resultado


def tabela_markdown(df, candidatos):
    md = ["# PROTÓTIPO: varredura do CART (ticket #4)", "",
          "Gerado por `src/prototipo_varredura_cart.py`. Células: F1 macro médio (DPA), k=5, semente 42.", ""]

    md += ["## max_depth (min_samples_leaf = 1)", "",
           "| max_depth | gini treino | gini validação | gini dif. | entropy treino | entropy validação | entropy dif. |",
           "|---|---|---|---|---|---|---|"]
    prof = df[df["min_samples_leaf"] == 1]
    for d in [*range(1, 17), 18, 20, 25, 30, None]:
        celulas = []
        for criterio in CRITERIOS:
            sel = prof[(prof["criterion"] == criterio) & (prof["max_depth"].isna() if d is None else prof["max_depth"] == d)]
            linha = sel.iloc[0]
            celulas += [celula(linha, "treino"), celula(linha, "validacao"), celula(linha, "diferenca")]
        md.append(f"| {rotulo_profundidade(d)} | " + " | ".join(celulas) + " |")
    md.append("")

    for criterio in CRITERIOS:
        for nome, titulo in [("validacao", "validação"), ("diferenca", "média das difs.")]:
            md += [f"## min_samples_leaf × max_depth, {criterio}: {titulo}", "",
                   "| min_samples_leaf | " + " | ".join(rotulo_profundidade(d) for d in PROFUNDIDADES_FOLHAS) + " |",
                   "|---" * (len(PROFUNDIDADES_FOLHAS) + 1) + "|"]
            for folha in FOLHAS:
                celulas = []
                for d in PROFUNDIDADES_FOLHAS:
                    prof_d = df["max_depth"].isna() if d is None else df["max_depth"] == d
                    linha = df[(df["criterion"] == criterio) & prof_d & (df["min_samples_leaf"] == folha)].iloc[0]
                    celulas.append(celula(linha, nome))
                md.append(f"| {folha} | " + " | ".join(celulas) + " |")
            md.append("")

    md += ["## Refinamento perto do pico: 10 maiores F1 de validação por criterion", ""]
    for criterio in CRITERIOS:
        sel = df[df["criterion"] == criterio].nlargest(10, "validacao_media")
        md += [f"### {criterio}", "", "| max_depth | min_samples_leaf | treino | validação | dif. |", "|---|---|---|---|---|"]
        for _, linha in sel.iterrows():
            md.append(f"| {rotulo_profundidade(linha['max_depth'])} | {int(linha['min_samples_leaf'])} | "
                      f"{celula(linha, 'treino')} | {celula(linha, 'validacao')} | {celula(linha, 'diferenca')} |")
        md.append("")

    md += ["## Conjuntos candidatos de U/E/O", ""]
    for nome, c in candidatos.items():
        marca = " (recomendado)" if nome == RECOMENDADO else ""
        res = c["resultados"]
        md += [f"### Conjunto {nome}{marca}: melhor CART = {c['melhor_cart']}", "",
               "| MODELO | U | E | O |", "|---|---|---|---|"]
        for chave, titulo in [("min_samples_leaf", "Min_samples_leaf"), ("max_depth", "Max_depth"),
                              ("criterion", "Criterion (gini, ent)")]:
            md.append(f"| {titulo} | " + " | ".join(str(res[r]["hiperparametrizacao"][chave]) for r in "UEO") + " |")
        for chave, titulo in [("treino", "TREINO f̄1 (DPA)"), ("validacao", "VALIDAÇÃO f̄1 (DPA)"),
                              ("diferenca", "MÉDIA DAS DIFS. (DPA)")]:
            md.append(f"| {titulo} | " + " | ".join(formatar(res[r][chave]) for r in "UEO") + " |")
        md.append("")
    return "\n".join(md)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    X, y = carregar_dataset()
    df = varrer(X, y)
    df.to_csv(SAIDA / "varredura.csv", index=False)
    grafico_profundidade(df)
    grafico_folhas(df)

    candidatos = avaliar_candidatos(X, y)
    (SAIDA / "candidatos.json").write_text(
        json.dumps({"recomendado": RECOMENDADO, **candidatos}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SAIDA / "tabela_varredura.md").write_text(tabela_markdown(df, candidatos) + "\n", encoding="utf-8")

    print(f"\nSaída em: {SAIDA.relative_to(RAIZ)}")
    for nome, c in candidatos.items():
        print(f"\nConjunto {nome}{' (recomendado)' if nome == RECOMENDADO else ''}: melhor CART = {c['melhor_cart']}")
        for rotulo, r in c["resultados"].items():
            print(f"  {rotulo} {r['hiperparametrizacao']}")
            print(f"    treino {formatar(r['treino'])}  validação {formatar(r['validacao'])}  dif {formatar(r['diferenca'])}")


if __name__ == "__main__":
    main()
