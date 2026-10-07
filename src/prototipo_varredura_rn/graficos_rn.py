"""PROTÓTIPO (descartável) — ticket #5. Gráficos e tabela da varredura da RN.

Lê saida/varredura_rn.jsonl (gerado por varredura_rn.py) e grava em saida/:
- rn_<eixo>.png: F1 macro médio de treino × validação (barras de erro = DPA), diferença
  treino-validação e tempo por avaliação, conforme a complexidade cresce da esquerda para
  a direita. Marcador vazado = houve ConvergenceWarning em algum fold.
- varredura_rn.md: tabela compacta de todas as avaliações.

Uso: uv run python src/prototipo_varredura_rn/graficos_rn.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from varredura_rn import AQUI, CONJUNTOS, ETAPAS, ler_cache  # noqa: I001 — põe src/ no sys.path
from validacao_cruzada import escolher_melhor, formatar

SAIDA = AQUI / "saida"
TABELA = SAIDA / "varredura_rn.md"

# Paleta de referência (slots 1 e 2) e tinta de texto.
COR_TREINO = "#2a78d6"
COR_VALIDACAO = "#eb6834"
COR_NEUTRA = "#8a8984"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
FUNDO = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
    "axes.edgecolor": TINTA_2,
    "axes.labelcolor": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "font.size": 9,
    "legend.frameon": False,
})


def topologia(h):
    return "[" + " ".join(str(n) for n in h["mlp__hidden_layer_sizes"]) + "]"


def rotulo_eixo(r):
    h = r["hiper"]
    eixo = r["eixo"]
    if eixo == "topologia":
        return f"{topologia(h)}\n{r['n_parametros']} p"
    if eixo == "ativacao":
        return h["mlp__activation"]
    if eixo == "alpha":
        return f"{h['mlp__alpha']:g}"
    if eixo == "solver_lr":
        s = h["mlp__solver"]
        return "lbfgs" if s == "lbfgs" else f"{s}\nlr={h['mlp__learning_rate_init']:g}"
    if eixo == "max_iter":
        return str(h["mlp__max_iter"])
    return rotulo_compacto(h)


def rotulo_compacto(h):
    """Topologia e só o que difere da BASE da varredura (relu, adam, lr=0.001, alpha=0.0001)."""
    partes = [topologia(h)]
    if h["mlp__activation"] != "relu":
        partes.append(h["mlp__activation"])
    if h["mlp__solver"] != "adam":
        partes.append(h["mlp__solver"])
    if h["mlp__solver"] != "lbfgs" and h["mlp__learning_rate_init"] != 0.001:
        partes.append(f"lr={h['mlp__learning_rate_init']:g}")
    if h["mlp__alpha"] != 0.0001:
        partes.append(f"α={h['mlp__alpha']:g}")
    if "mlp__n_iter_no_change" in h:
        partes.append(f"paciência={h['mlp__n_iter_no_change']}")
    return "\n".join(partes)


def grafico(registros, titulo, xlabel, arquivo, rotulos=None, piso=None, grupos=None, legenda="lower right"):
    """Três painéis com o mesmo eixo x: F1 de treino × validação, diferença e tempo.

    Valores de F1 abaixo de `piso` (ex.: RN que colapsa na classe mais frequente, 0.12061) são
    desenhados no piso como triângulo para baixo, com o valor escrito, para não achatar o resto.
    `grupos` (tamanho de cada grupo) liga os pontos só dentro de cada grupo e separa os grupos.
    """
    rotulos = rotulos or [rotulo_eixo(r) for r in registros]
    x = range(len(registros))
    grupos = grupos or [len(registros)]
    fig, (ax_f1, ax_dif, ax_t) = plt.subplots(
        3, 1, figsize=(max(6.5, 0.62 * len(registros) + 2), 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3, 1.1]},
    )
    for chave, cor, marcador, nome in [
        ("treino", COR_TREINO, "o", "treino"),
        ("validacao", COR_VALIDACAO, "s", "validação"),
    ]:
        medias = [r["resultado"][chave]["media"] for r in registros]
        dpas = [r["resultado"][chave]["dpa"] for r in registros]
        visiveis = [m if piso is None else max(m, piso) for m in medias]
        inicio = 0
        for n in grupos:
            fatia = slice(inicio, inicio + n)
            ax_f1.errorbar(list(x)[fatia], visiveis[fatia], yerr=dpas[fatia], color=cor, lw=2, capsize=3,
                           zorder=2, label=nome if inicio == 0 else None)
            inicio += n
            if inicio < len(registros):
                for ax in (ax_f1,):
                    ax.axvline(inicio - 0.5, color="#c9c8c2", lw=1, zorder=1)
        for i, r in enumerate(registros):
            convergiu = r["avisos_convergencia"] == 0
            abaixo = piso is not None and medias[i] < piso
            ax_f1.plot(i, visiveis[i], "v" if abaixo else marcador, ms=8, color=cor,
                       mfc=cor if convergiu else FUNDO, mew=2, zorder=3)
            if abaixo and chave == "validacao":
                ax_f1.annotate(f"{medias[i]:.3f}\n(abaixo)", (i, piso), textcoords="offset points",
                               xytext=(0, 9), ha="center", fontsize=7, color=TINTA_2)
    nao_conv = [i for i, r in enumerate(registros) if r["avisos_convergencia"]]
    for i in nao_conv:
        r = registros[i]
        ax_f1.annotate(f"{r['avisos_convergencia']}/5\nnão conv.", (i, r["resultado"]["treino"]["media"]),
                       textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color=TINTA_2)
    if nao_conv:
        ax_f1.plot([], [], "o", ms=8, color=TINTA_2, mfc=FUNDO, mew=2, label="ConvergenceWarning em algum fold")
    if piso is not None:
        ax_f1.set_ylim(bottom=piso - 0.01)
    ax_f1.set_ylabel("F1 macro médio")
    ax_f1.set_title(titulo, loc="left", color=TINTA, fontsize=11)
    ax_f1.legend(loc=legenda, fontsize=8)

    difs = [r["resultado"]["diferenca"]["media"] for r in registros]
    dpas_dif = [r["resultado"]["diferenca"]["dpa"] for r in registros]
    ax_dif.bar(x, difs, yerr=dpas_dif, width=0.6, color=COR_NEUTRA, ecolor=TINTA_2, capsize=2)
    ax_dif.set_ylabel("dif. treino-\nvalidação")
    for i, (d, e) in enumerate(zip(difs, dpas_dif)):
        ax_dif.annotate(f"{d:.3f}", (i, d + e), textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=6.5, color=TINTA_2)
    ax_dif.set_ylim(top=max(d + e for d, e in zip(difs, dpas_dif)) * 1.25 + 0.002)

    ax_t.bar(x, [r["tempo_s"] for r in registros], width=0.6, color=COR_NEUTRA)
    ax_t.set_ylabel("tempo (s), 5 folds\n(12 em paralelo)")
    ax_t.set_xticks(list(x), rotulos, fontsize=7.5)
    ax_t.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(SAIDA / arquivo, dpi=130)
    plt.close(fig)
    print(f"Gráfico: {(SAIDA / arquivo).relative_to(AQUI.parent.parent)}")


def linha_tabela(r):
    h = r["hiper"]
    res = r["resultado"]
    cel = lambda e: f"{e['media']:.5f} ({e['dpa']:.5f})"  # noqa: E731 — mesmo formato de formatar()
    lr = "—" if h["mlp__solver"] == "lbfgs" else f"{h['mlp__learning_rate_init']:g}"
    extras = ", ".join(f"{k.removeprefix('mlp__')}={v}" for k, v in h.items()
                       if k not in {"mlp__hidden_layer_sizes", "mlp__activation", "mlp__solver",
                                    "mlp__learning_rate_init", "mlp__alpha", "mlp__max_iter"})
    epocas = r["epocas_por_fold"]
    return (
        f"| {r['eixo']} | {topologia(h)} | {h['mlp__activation']} | {h['mlp__solver']} | {lr} | "
        f"{h['mlp__alpha']:g} | {h['mlp__max_iter']} | {extras} | {cel(res['treino'])} | "
        f"{cel(res['validacao'])} | {cel(res['diferenca'])} | {r['tempo_s']:.1f} | "
        f"{5 - r['avisos_convergencia']}/5 | {min(epocas)}–{max(epocas)} |"
    )


def main():
    cache = ler_cache()
    por_etapa = {nome: [cache[c["id"]] for c in etapa() if c["id"] in cache] for nome, etapa in ETAPAS.items()}

    grossa = por_etapa["grossa"]
    def eixo(nome):
        return [r for r in grossa if r["eixo"] == nome]

    topo = sorted(eixo("topologia"), key=lambda r: r["n_parametros"])
    if topo:
        grafico(topo, "RN: topologia (relu, adam, lr=0.001, alpha=0.0001, max_iter=2000)",
                "topologia (neurônios por camada oculta) e nº de parâmetros, em ordem crescente",
                "rn_topologia.png", piso=0.85)
    alpha = sorted(eixo("alpha"), key=lambda r: -r["hiper"]["mlp__alpha"])
    if alpha:
        grafico(alpha, "RN: regularização L2 (alpha), topologia [64 64], relu, adam, max_iter=2000",
                "alpha (decrescente: menos regularização → mais complexidade efetiva)", "rn_alpha.png")
    if eixo("max_iter"):
        grafico(eixo("max_iter"), "RN padrão ([100], relu, adam, lr=0.001, alpha=0.0001): limite de épocas",
                "max_iter (200 = padrão do scikit-learn, item 23)", "rn_max_iter.png", piso=0.85)
    ativ_solver = eixo("ativacao") + eixo("solver_lr")
    if ativ_solver:
        grafico(ativ_solver, "RN [32 16]: função de ativação (relu/adam/lr=0.001) e solver/learning rate (relu)",
                "função de ativação  |  solver e learning rate", "rn_ativacao_solver.png", piso=0.85)
    if por_etapa["subajuste"]:
        u = sorted(por_etapa["subajuste"], key=lambda r: r["resultado"]["validacao"]["media"])
        grafico(u, "RN: busca de uma U (subajuste), em ordem de F1 de validação; base relu, adam, lr=0.001, α=0.0001",
                "hiperparametrização (só o que difere da base)", "rn_subajuste.png")
    if por_etapa["sobreajuste"]:
        o = sorted(por_etapa["sobreajuste"], key=lambda r: r["resultado"]["diferenca"]["media"])
        grafico(o, "RN: busca de uma O (sobreajuste), em ordem de diferença; base relu, adam, lr=0.001",
                "hiperparametrização (só o que difere da base)", "rn_sobreajuste.png", piso=0.85)

    # Conjuntos candidatos: procura cada hiperparametrização no cache, venha de que etapa vier.
    por_hiper = {json.dumps(r["hiper"], sort_keys=True): r for r in cache.values()}
    candidatos, rotulos = [], []
    for nome, conjunto in CONJUNTOS.items():
        for rotulo, h in conjunto.items():
            candidatos.append(por_hiper[json.dumps(h, sort_keys=True)])
            rotulos.append(f"{nome}·{rotulo}\n{rotulo_compacto(h)}")
    grafico(candidatos, "RN: conjuntos candidatos a U/E/O (A = recomendado)", "conjunto · hiperparametrização",
            "rn_candidatos.png", rotulos=rotulos, piso=0.6, grupos=[3] * len(CONJUNTOS),
            legenda="lower center")

    linhas = [
        "# Varredura da RN (PROTÓTIPO, ticket #5)",
        "",
        "F1 macro médio (DPA) em k=5 folds estratificados (semente 42), via `avaliar`. "
        "Tempo = avaliação inteira (5 fits), medido com 12 avaliações em paralelo e 1 thread cada. "
        "Conv. = folds sem ConvergenceWarning. Épocas = n_iter_ mínimo–máximo entre os folds. "
        "Learning rate \"—\" = lbfgs, que ignora `learning_rate_init`.",
        "",
        "Sozinho e com 1 thread (`src/treinar_rn.py`), o conjunto A leva 16 s (U), 18 s (E) e 144 s (O), "
        "com os mesmos números da varredura. Com as threads padrão do BLAS, a O passou de 10 min sem terminar.",
        "",
    ]
    linhas += ["## Conjuntos candidatos a U/E/O", ""]
    for nome, conjunto in CONJUNTOS.items():
        regs = {rotulo: por_hiper[json.dumps(h, sort_keys=True)] for rotulo, h in conjunto.items()}
        melhor = escolher_melhor({rotulo: r["resultado"] for rotulo, r in regs.items()})
        hs = [r["hiper"] for r in regs.values()]
        lr = ["— (lbfgs)" if h["mlp__solver"] == "lbfgs" else f"{h['mlp__learning_rate_init']:g}" for h in hs]
        linhas += [
            f"### Conjunto {nome} (melhor RN por `escolher_melhor`: {melhor})",
            "",
            "| MODELO | U | E | O |",
            "|---|---|---|---|",
            "| Topologia | " + " | ".join(topologia(h) for h in hs) + " |",
            "| Função de ativação | " + " | ".join(h["mlp__activation"] for h in hs) + " |",
            "| Learning rate | " + " | ".join(lr) + " |",
            "| solver | " + " | ".join(h["mlp__solver"] for h in hs) + " |",
            "| alpha | " + " | ".join(f"{h['mlp__alpha']:g}" for h in hs) + " |",
            "| max_iter | " + " | ".join(str(h["mlp__max_iter"]) for h in hs) + " |",
            "| TREINO f̄1 (DPA) | " + " | ".join(formatar(r["resultado"]["treino"]) for r in regs.values()) + " |",
            "| VALIDAÇÃO f̄1 (DPA) | " + " | ".join(formatar(r["resultado"]["validacao"]) for r in regs.values()) + " |",
            "| MÉDIA DAS DIFS. (DPA) | " + " | ".join(formatar(r["resultado"]["diferenca"]) for r in regs.values()) + " |",
            "| folds sem ConvergenceWarning | " + " | ".join(f"{5 - r['avisos_convergencia']}/5" for r in regs.values()) + " |",
            "| épocas/iterações por fold | " + " | ".join(f"{min(r['epocas_por_fold'])}–{max(r['epocas_por_fold'])}" for r in regs.values()) + " |",
            "| tempo da avaliação (s, em paralelo) | " + " | ".join(f"{r['tempo_s']:.0f}" for r in regs.values()) + " |",
            "",
        ]
    for nome, registros in por_etapa.items():
        if not registros:
            continue
        linhas += [
            f"## Etapa {nome}",
            "",
            "| eixo | topologia | ativação | solver | lr | alpha | max_iter | outros | treino | validação | "
            "diferença | tempo (s) | conv. | épocas |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            *[linha_tabela(r) for r in registros],
            "",
        ]
    TABELA.write_text("\n".join(linhas), encoding="utf-8")
    print(f"Tabela: {TABELA.relative_to(AQUI.parent.parent)}")


if __name__ == "__main__":
    main()
