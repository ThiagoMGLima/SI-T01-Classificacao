"""PROTÓTIPO (descartável) — ticket #6 "Definir como o relatório PDF é gerado".

Parte comum aos candidatos: lê os números e os transforma em textos prontos para as
8 tabelas do enunciado. Os candidatos só diferem na diagramação (como viram PDF), então
a comparação entre eles é só visual.

Tabela 1 vem do arquivo real `resultados/tabela1_dataset.json`; Tabelas 2 a 8 vêm dos
números FICTÍCIOS de `dados_ficticios/` (ver gerar_dados_ficticios.py).
"""

import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SAIDA = AQUI / "saida"

# Títulos das tabelas, copiados do enunciado (seção RELATÓRIO).
TITULOS = {
    1: "1) DATASET DE TREINAMENTO/VALIDAÇÃO",
    2: "2) HIPERPARAMETRIZAÇÕES CART",
    3: "3) RESULTADOS DOS MODELOS CART",
    4: "4) HIPERPARAMETRIZAÇÕES REDE NEURAL",
    5: "5) RESULTADOS DOS MODELOS RN",
    6: "6) MELHOR MODELO CART X MELHOR MODELO RN",
    7: "7) RESULTADOS DO TESTE CEGO MELHOR CART X MELHOR RN",
    8: "8) MATRIZ DE CONFUSÃO DO MELHOR CART e MATRIZ DE CONFUSÃO DA MELHOR RN",
}
SUBTITULO_8 = "G=verde, Y=amarelo, R=vermelho, B=preto"
LETRAS = ["G", "Y", "R", "B"]

# Cores de cabeçalho amostradas do PDF do enunciado.
COR = {1: "#E5DFEC", 2: "#DAEEF3", 3: "#DAEEF3", 4: "#FDE9D9", 5: "#FDE9D9", 6: "#B2A1C7", 7: "#B2A1C7", 8: "#B2A1C7"}

# Linhas das Tabelas 3, 5 e 6. Cada candidato monta o rótulo com a sua notação de f̄1:
# (antes do f̄1, depois do f̄1); MÉDIA DAS DIFS. não tem f̄1.
ROTULOS_F1 = {
    "treino": ("TREINO ", " (DPA)"),
    "validacao": ("VALIDAÇÃO ", " (DPA)"),
    "diferenca": ("MÉDIA DAS DIFS. (DPA)", None),
}


def ler(caminho):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def num(x, casas=5):
    return f"{x:.{casas}f}"


def valor_hiper(v):
    if v is None:
        return "None"
    if isinstance(v, list):  # topologia: [32 16]
        return "[" + " ".join(str(n) for n in v) + "]"
    return str(v)


def resultados_cv(res, nomes):
    """Linhas das Tabelas 3/5/6: {chave: [(f̄1, dpa), ...]} na ordem de `nomes`."""
    return {chave: [(num(res[n][chave]["media"]), num(res[n][chave]["dpa"])) for n in nomes] for chave in ROTULOS_F1}


def montar_relatorio():
    t1 = ler(RAIZ / "resultados" / "tabela1_dataset.json")["tabela_1"]
    cart = ler(AQUI / "dados_ficticios" / "cart_validacao_cruzada.json")
    rn = ler(AQUI / "dados_ficticios" / "rn_validacao_cruzada.json")
    teste = ler(AQUI / "dados_ficticios" / "teste_cego.json")

    hc, hr = cart["hiperparametrizacoes"], rn["hiperparametrizacoes"]
    ueo = ["U", "E", "O"]
    melhor_cart, melhor_rn = cart["melhor"], rn["melhor"]

    return {
        1: {
            "colunas": ["MODELO", "VALOR"],
            "linhas": [
                *[[f"Vítimas tri={cor}", str(n)] for cor, n in t1["vitimas_por_classe"].items()],
                ["Idade das vítimas", num(t1["idade_media"], 2)],
                ["Desvio padrão amostral da idade", num(t1["dpa_idade"], 2)],
                ["ruído", num(t1["ruido"], 2)],
            ],
        },
        2: {
            "colunas": ["MODELO", *ueo],
            "linhas": [
                ["Min_samples_leaf", *[valor_hiper(hc[n]["min_samples_leaf"]) for n in ueo]],
                ["Max_depth", *[valor_hiper(hc[n]["max_depth"]) for n in ueo]],
                ["Criterion (gini, ent)", *[valor_hiper(hc[n]["criterion"]) for n in ueo]],
            ],
        },
        3: {"colunas": ["MODELO", *ueo], "linhas": resultados_cv(cart["resultados"], ueo)},
        4: {
            "colunas": ["MODELO", *ueo],
            "linhas": [
                ["Topologia", *[valor_hiper(hr[n]["hidden_layer_sizes"]) for n in ueo]],
                ["Função de ativação", *[valor_hiper(hr[n]["activation"]) for n in ueo]],
                ["Learning rate", *[valor_hiper(hr[n]["learning_rate_init"]) for n in ueo]],
                ["solver", *[valor_hiper(hr[n]["solver"]) for n in ueo]],
            ],
        },
        5: {"colunas": ["MODELO", *ueo], "linhas": resultados_cv(rn["resultados"], ueo)},
        6: {
            "colunas": ["MODELO", f"Melhor CART ({melhor_cart})", f"Melhor RN ({melhor_rn})"],
            "linhas": {
                chave: [resultados_cv(cart["resultados"], [melhor_cart])[chave][0],
                        resultados_cv(rn["resultados"], [melhor_rn])[chave][0]]
                for chave in ROTULOS_F1
            },
        },
        7: {
            "colunas": ["MODELO", f"Melhor CART ({teste['cart']['hiperparametrizacao']})",
                        f"Melhor RN ({teste['rn']['hiperparametrizacao']})"],
            "linhas": [
                [rotulo, num(teste["cart"][chave]), num(teste["rn"][chave])]
                for rotulo, chave in [
                    ("MÉDIA PRECISÃO (macro)", "precisao_macro"),
                    ("MÉDIA RECALL (macro)", "recall_macro"),
                    ("F1 SCORE (macro)", "f1_macro"),
                    ("ACURÁCIA", "acuracia"),
                ]
            ],
        },
        8: {
            "CART": [[str(v) for v in linha] for linha in teste["cart"]["matriz_confusao"]],
            "RN": [[str(v) for v in linha] for linha in teste["rn"]["matriz_confusao"]],
        },
    }
