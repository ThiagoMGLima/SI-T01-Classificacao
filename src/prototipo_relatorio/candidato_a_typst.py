"""PROTÓTIPO (descartável) — ticket #6, candidato A: Typst.

O Python só lê os números (comum.py) e passa para o template `candidato_a_typst/relatorio.typ`,
que faz toda a diagramação. O compilador vem no pacote PyPI `typst` (binário próprio, sem
LaTeX nem dependência de sistema).

Uso: uv run --with typst python src/prototipo_relatorio/candidato_a_typst.py
"""

import json

import typst

from comum import AQUI, COR, LETRAS, ROTULOS_F1, SAIDA, SUBTITULO_8, TITULOS, montar_relatorio


def main():
    relatorio = montar_relatorio()
    dados = {
        "titulos": {str(n): t for n, t in TITULOS.items()},
        "cores": {str(n): c for n, c in COR.items()},
        "rotulos_f1": ROTULOS_F1,
        "letras": LETRAS,
        "subtitulo_8": SUBTITULO_8,
        **{f"t{n}": tabela for n, tabela in relatorio.items()},
    }
    SAIDA.mkdir(exist_ok=True)
    destino = SAIDA / "candidato_a_typst.pdf"
    typst.compile(
        str(AQUI / "candidato_a_typst" / "relatorio.typ"),
        output=str(destino),
        sys_inputs={"dados": json.dumps(dados, ensure_ascii=False)},
    )
    print("gravado", destino)


if __name__ == "__main__":
    main()
