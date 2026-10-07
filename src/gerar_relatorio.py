"""Gera o relatório PDF com as 8 tabelas do enunciado e monta o pacote de entrega em entrega/.

Última etapa. Lê só os números que os scripts anteriores salvaram em resultados/ (Tabela 1 de
gerar_dataset.py, 2 e 3 de treinar_cart.py, 4 e 5 de treinar_rn.py, 6 a 8 de teste_cego.py),
formata-os como texto e os passa em JSON (sys.inputs) ao template Typst relatorio/relatorio.typ,
que faz toda a diagramação (item 24 de observacoes.md). O PDF tem somente as 8 tabelas da seção
RELATÓRIO do enunciado, sem nenhum texto adicional (item 13). Nunca lê os datasets.

Depois monta em entrega/ o que a seção ENTREGA do enunciado pede:
- relatorio_T01.pdf: o relatório;
- melhor_cart.joblib e melhor_rn.joblib: cópias dos modelos de modelos/;
- codigo_fonte_T01.zip: src/ (scripts e template), vendor/victsim3/, pyproject.toml, uv.lock e
  .python-version, sem os datasets.
O PDF e o zip saem idênticos byte a byte a cada execução: o PDF não leva data de criação, e o zip
usa ordem de arquivos, datas e permissões fixas.

Uso: uv run python src/gerar_relatorio.py
"""

import json
import shutil
import zipfile
from pathlib import Path

import typst

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
TABELA_1 = RESULTADOS / "tabela1_dataset.json"
TABELAS_2_3 = RESULTADOS / "tabelas2_3_cart.json"
TABELAS_4_5 = RESULTADOS / "tabelas4_5_rn.json"
TABELAS_6_7_8 = RESULTADOS / "tabelas6_7_8_teste_cego.json"
TEMPLATE = RAIZ / "src" / "relatorio" / "relatorio.typ"

ENTREGA = RAIZ / "entrega"
RELATORIO = ENTREGA / "relatorio_T01.pdf"
CODIGO_FONTE = ENTREGA / "codigo_fonte_T01.zip"

# Modelos salvos por teste_cego.py, já com os nomes da seção ENTREGA do enunciado (item 45).
MODELOS = [RAIZ / "modelos" / "melhor_cart.joblib", RAIZ / "modelos" / "melhor_rn.joblib"]

# Código-fonte do zip, relativo à raiz do repositório. Os datasets ficam de fora: o de
# treino/validação sai de novo, idêntico, de src/gerar_dataset.py (semente 42), e o teste cego é o
# 1300v do VictSim3, que o professor já tem (a origem está em vendor/victsim3/README.md).
CODIGO = ["src", "vendor/victsim3", "pyproject.toml", "uv.lock", ".python-version"]
PASTA_NO_ZIP = "codigo_fonte_T01"
DATA_FIXA = (1980, 1, 1, 0, 0, 0)  # a menor data do formato zip

UEO = ["U", "E", "O"]
CORES = ["verde", "amarelo", "vermelho", "preto"]
LETRAS = ["G", "Y", "R", "B"]  # classes de triagem na ordem de tri (0 a 3), como na Tabela 8

# "f̄1" (f com mácron, U+0304): o template o troca pelo símbolo em fonte matemática.
F1 = "f̄" + "1"

# Títulos e rótulos de linha, copiados da seção RELATÓRIO do enunciado.
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

LINHAS_TABELA_2 = {
    "min_samples_leaf": "Min_samples_leaf",
    "max_depth": "Max_depth",
    "criterion": "Criterion (gini, ent)",
}
LINHAS_F1 = {  # Tabelas 3, 5 e 6
    "treino": f"TREINO {F1} (DPA)",
    "validacao": f"VALIDAÇÃO {F1} (DPA)",
    "diferenca": "MÉDIA DAS DIFS. (DPA)",
}
LINHAS_TABELA_4 = {
    "hidden_layer_sizes": "Topologia",
    "activation": "Função de ativação",
    "learning_rate_init": "Learning rate",
    "solver": "solver",
    # Linhas extras, que o enunciado permite: completam a hiperparametrização (item 38).
    "alpha": "alpha",
    "max_iter": "max_iter",
}
LINHAS_TABELA_7 = {
    "precisao_macro": "MÉDIA PRECISÃO (macro)",
    "recall_macro": "MÉDIA RECALL (macro)",
    "f1_macro": "F1 SCORE (macro)",
    "acuracia": "ACURÁCIA",
}


def ler_json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def hiperparametro(valor, ausente):
    """Célula das Tabelas 2 e 4: topologia como [128 128] e float sem zeros à direita (0.001, 0).

    `ausente` é o texto de um null do JSON: "None" no max_depth sem limite da Tabela 2 (item 31) e
    "—" no learning rate que o lbfgs ignora, na Tabela 4 (item 41).
    """
    if valor is None:
        return ausente
    if isinstance(valor, list):  # neurônios por camada oculta
        return "[" + " ".join(str(n) for n in valor) + "]"
    if isinstance(valor, float):
        return f"{valor:g}"
    return str(valor)


def media_dpa(estatistica):
    """Célula das Tabelas 3, 5 e 6 como [média, DPA], com 5 casas; o template a mostra como
    "0.88212 (0.01074)", o formato de formatar em validacao_cruzada.py."""
    return [f"{estatistica['media']:.5f}", f"{estatistica['dpa']:.5f}"]


def linhas_f1(resultados, rotulos):
    """Linhas das Tabelas 3, 5 e 6: treino, validação e diferença, uma coluna por rótulo."""
    return [[nome, *[media_dpa(resultados[r][chave]) for r in rotulos]] for chave, nome in LINHAS_F1.items()]


def conferir(cart, rn, teste):
    """Confere que os JSON das Tabelas 2 a 8 vêm da mesma sequência de execuções.

    Os rótulos do melhor CART e da melhor RN e a Tabela 6 do teste cego têm de ser os das
    Tabelas 3 e 5, e a Tabela 8 tem de estar na ordem G, Y, R, B, com a classe real nas linhas.
    """
    for nome, resultados, tabela in [("cart", cart, "tabela_3"), ("rn", rn, "tabela_5")]:
        melhor = resultados[f"melhor_{nome}"]
        if teste[f"melhor_{nome}"] != melhor:
            raise SystemExit(f"{nome}: o melhor do teste cego ({teste[f'melhor_{nome}']}) difere de {melhor}")
        if teste["tabela_6"][nome] != resultados[tabela][melhor]:
            raise SystemExit(f"{nome}: a Tabela 6 difere da {tabela}[{melhor}]")
    t8 = teste["tabela_8"]
    if (t8["ordem"], t8["linhas"], t8["colunas"]) != (LETRAS, "real", "predito"):
        raise SystemExit(f"Tabela 8 fora da ordem {LETRAS} (linhas: real; colunas: predito)")


def montar_tabelas():
    """Textos das 8 tabelas, prontos para o template: {"t1": ..., ..., "t8": ...}."""
    t1 = ler_json(TABELA_1)["tabela_1"]
    cart, rn, teste = ler_json(TABELAS_2_3), ler_json(TABELAS_4_5), ler_json(TABELAS_6_7_8)
    conferir(cart, rn, teste)
    melhores = [f"Melhor CART ({teste['melhor_cart']})", f"Melhor RN ({teste['melhor_rn']})"]
    t8 = teste["tabela_8"]
    return {
        "t1": {
            "titulo": TITULOS[1],
            "colunas": ["MODELO", "VALOR"],
            "linhas": [
                *[[f"Vítimas tri={cor}", str(t1["vitimas_por_classe"][cor])] for cor in CORES],
                ["Idade das vítimas", f"{t1['idade_media']:.2f}"],
                # Rótulo do enunciado, embora o valor seja o desvio populacional (itens 9 e 26).
                ["Desvio padrão amostral da idade", f"{t1['dpa_idade']:.2f}"],
                ["ruído", f"{t1['ruido']:g}"],
            ],
        },
        "t2": {
            "titulo": TITULOS[2],
            "colunas": ["MODELO", *UEO],
            "linhas": [
                [nome, *[hiperparametro(cart["tabela_2"][r][chave], "None") for r in UEO]]
                for chave, nome in LINHAS_TABELA_2.items()
            ],
        },
        "t3": {"titulo": TITULOS[3], "colunas": ["MODELO", *UEO], "linhas": linhas_f1(cart["tabela_3"], UEO)},
        "t4": {
            "titulo": TITULOS[4],
            "colunas": ["MODELO", *UEO],
            "linhas": [
                [nome, *[hiperparametro(rn["tabela_4"][r][chave], "—") for r in UEO]]
                for chave, nome in LINHAS_TABELA_4.items()
            ],
        },
        "t5": {"titulo": TITULOS[5], "colunas": ["MODELO", *UEO], "linhas": linhas_f1(rn["tabela_5"], UEO)},
        "t6": {
            "titulo": TITULOS[6],
            "colunas": ["MODELO", *melhores],
            "linhas": linhas_f1(teste["tabela_6"], ["cart", "rn"]),
        },
        "t7": {
            "titulo": TITULOS[7],
            "colunas": ["MODELO", *melhores],
            "linhas": [
                [nome, *[f"{teste['tabela_7'][m][chave]:.5f}" for m in ("cart", "rn")]]
                for chave, nome in LINHAS_TABELA_7.items()
            ],
        },
        "t8": {
            "titulo": TITULOS[8],
            "subtitulo": SUBTITULO_8,
            "colunas": ["CART", "RN"],
            "letras": t8["ordem"],
            "real": t8["linhas"],
            "predito": t8["colunas"],
            "matrizes": [[[str(n) for n in linha] for linha in t8[m]] for m in ("cart", "rn")],
        },
    }


def gerar_pdf(tabelas, destino):
    """Compila o template com as tabelas e grava o PDF.

    Para em qualquer aviso do Typst: sem as fontes Carlito e Liberation Mono, por exemplo, ele só
    avisa e troca pela fonte padrão (item 25).
    """
    pdf, avisos = typst.compile_with_warnings(
        str(TEMPLATE), sys_inputs={"dados": json.dumps(tabelas, ensure_ascii=False)}
    )
    if avisos:
        raise SystemExit("O Typst emitiu avisos:\n" + "\n".join(f"- {a.message}" for a in avisos))
    destino.write_bytes(pdf)


def arquivos_do_codigo():
    """Arquivos do código-fonte do zip, em ordem fixa e sem os caches do Python."""
    for item in CODIGO:
        caminho = RAIZ / item
        if caminho.is_dir():
            yield from sorted(p for p in caminho.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        else:
            yield caminho


def montar_zip(destino):
    """Zip do código-fonte, dentro da pasta PASTA_NO_ZIP. Data e permissões fixas em todos os
    arquivos, para que duas execuções deem o mesmo zip byte a byte. Devolve os caminhos no zip."""
    nomes = []
    with zipfile.ZipFile(destino, "w") as zf:
        for arquivo in arquivos_do_codigo():
            info = zipfile.ZipInfo(f"{PASTA_NO_ZIP}/{arquivo.relative_to(RAIZ).as_posix()}", date_time=DATA_FIXA)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16  # rw-r--r--
            zf.writestr(info, arquivo.read_bytes(), compresslevel=9)
            nomes.append(info.filename)
    return nomes


def main():
    ENTREGA.mkdir(exist_ok=True)
    gerar_pdf(montar_tabelas(), RELATORIO)
    for modelo in MODELOS:
        shutil.copyfile(modelo, ENTREGA / modelo.name)
    nomes = montar_zip(CODIGO_FONTE)

    print(f"Pacote de entrega em {ENTREGA.relative_to(RAIZ)}/:")
    for arquivo in sorted(ENTREGA.iterdir()):
        print(f"  {arquivo.name:<24} {arquivo.stat().st_size:>8} bytes")
    print(f"\n{CODIGO_FONTE.name} ({len(nomes)} arquivos):")
    for nome in nomes:
        print(f"  {nome}")


if __name__ == "__main__":
    main()
