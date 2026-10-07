"""PROTÓTIPO (descartável) — ticket #6, candidato B: HTML + CSS convertido em PDF pelo WeasyPrint.

O Python monta uma página HTML (f-strings, sem motor de templates) com os números do
comum.py; o WeasyPrint (pacote PyPI, usa Pango do sistema) converte em PDF. O mesmo HTML
também é gravado em `saida/`, e pode ser convertido pelo Chrome sem instalar nada:
  google-chrome --headless --no-pdf-header-footer --print-to-pdf=saida/x.pdf saida/candidato_b_html.html

Uso: uv run --with weasyprint python src/prototipo_relatorio/candidato_b_html.py
"""

from html import escape

import weasyprint

from comum import COR, LETRAS, ROTULOS_F1, SAIDA, SUBTITULO_8, TITULOS, montar_relatorio

# f̄1: f itálico com traço em cima (Caladea ≈ Cambria, a fonte de fórmulas do enunciado) e 1 subscrito.
F1 = '<span class="f1"><i>f</i><sub>1</sub></span>'

CSS = """
@page { size: A4; margin: 2.5cm 3cm; }
body { font-family: Carlito, Calibri, sans-serif; font-size: 11pt; }
h2 { font-size: 11pt; margin: 1.4em 0 0.15em; }
h2 + p { font-weight: bold; margin: 0 0 0.15em; }
section { break-inside: avoid; }
table { border-collapse: collapse; }
th, td { border: 0.6pt solid black; padding: 0.5pt 5pt; line-height: 1.25; text-align: center; }
th { font-weight: bold; }
td.esq { text-align: left; }
td.rotulo { font-size: 8.5pt; }
small { font-size: 8.5pt; }
.f1 { margin: 0 0.1em; }
.f1 i { font-family: Caladea, Cambria, serif; display: inline-block; line-height: 0.95; border-top: 0.5pt solid; padding: 0 0.05em 0 0.15em; }
.f1 sub { font-family: Caladea, Cambria, serif; font-size: 70%; }
/* Tabela 8: duas células com a matriz em fonte monoespaçada, como no enunciado */
table.t8 th { font-family: "Liberation Mono", "Courier New", monospace; width: 6.6cm; }
table.t8 td { text-align: left; padding: 3pt 8pt; }
table.mc { font-family: "Liberation Mono", "Courier New", monospace; font-size: 10pt; }
table.mc td { border: none; padding: 0.5pt 0 0.5pt 0.9em; text-align: right; }
table.mc td.l { text-align: left; padding-left: 0.6em; }
table.mc em { font-style: italic; }
"""


def tabela(n, colunas, linhas, larguras, esq=True, rotulo_pequeno=False):
    """`linhas`: listas de células já em HTML."""
    cab = "".join(f'<th style="background:{COR[n]}">{escape(c)}</th>' for c in colunas)
    corpo = ""
    for linha in linhas:
        classe = "rotulo" if rotulo_pequeno else ("esq" if esq else "")
        corpo += f'<tr><td class="{classe}">{linha[0]}</td>' + "".join(f"<td>{c}</td>" for c in linha[1:]) + "</tr>"
    cols = "".join(f'<col style="width:{w}">' for w in larguras)
    return f"<table><colgroup>{cols}</colgroup><tr>{cab}</tr>{corpo}</table>"


def linhas_f1(t):
    linhas = []
    for chave, (antes, depois) in ROTULOS_F1.items():
        rotulo = escape(antes) if depois is None else f"{escape(antes)}{F1}{escape(depois)}"
        linhas.append([rotulo, *[f"{m} <small>({d})</small>" for m, d in t["linhas"][chave]]])
    return linhas


def texto(linhas):
    return [[escape(c) for c in linha] for linha in linhas]


def matriz(m):
    linhas = ['<tr><td class="l"></td><td class="l"></td>' + "".join(f"<td>{l}</td>" for l in LETRAS) + "</tr>"]
    for i, l in enumerate(LETRAS):
        real = "<em>real</em>" if i == 1 else ""
        linhas.append(f'<tr><td class="l">{real}</td><td class="l">{l}</td>' + "".join(f"<td>{v}</td>" for v in m[i]) + "</tr>")
    linhas.append('<tr><td class="l"></td><td class="l"></td><td colspan="4" style="text-align:center"><em>predito</em></td></tr>')
    return f'<table class="mc">{"".join(linhas)}</table>'


def secao(n, conteudo, subtitulo=""):
    sub = f"<p>{escape(subtitulo)}</p>" if subtitulo else ""
    return f"<section><h2>{escape(TITULOS[n])}</h2>{sub}{conteudo}</section>"


def main():
    r = montar_relatorio()
    larg_res = ["4.3cm", "3.4cm", "4cm", "4cm"]
    corpo = "".join([
        secao(1, tabela(1, r[1]["colunas"], texto(r[1]["linhas"]), ["6.6cm", "3.6cm"])),
        secao(2, tabela(2, r[2]["colunas"], texto(r[2]["linhas"]), ["4.3cm", "2.8cm", "2.8cm", "3.2cm"])),
        secao(3, tabela(3, r[3]["colunas"], linhas_f1(r[3]), larg_res, rotulo_pequeno=True)),
        secao(4, tabela(4, r[4]["colunas"], texto(r[4]["linhas"]), ["4.3cm", "4cm", "4cm", "4cm"])),
        secao(5, tabela(5, r[5]["colunas"], linhas_f1(r[5]), larg_res, rotulo_pequeno=True)),
        secao(6, tabela(6, r[6]["colunas"], linhas_f1(r[6]), ["4.3cm", "4cm", "3.4cm"], rotulo_pequeno=True)),
        secao(7, tabela(7, r[7]["colunas"], texto(r[7]["linhas"]), ["3.7cm", "4cm", "4cm"], rotulo_pequeno=True)),
        secao(8, '<table class="t8"><tr>'
                 + "".join(f'<th style="background:{COR[8]}">{n}</th>' for n in ("CART", "RN"))
                 + f'</tr><tr><td>{matriz(r[8]["CART"])}</td><td>{matriz(r[8]["RN"])}</td></tr></table>',
              SUBTITULO_8),
    ])
    html = f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{corpo}</body></html>'

    SAIDA.mkdir(exist_ok=True)
    (SAIDA / "candidato_b_html.html").write_text(html, encoding="utf-8")
    destino = SAIDA / "candidato_b_html.pdf"
    weasyprint.HTML(string=html).write_pdf(destino)
    print("gravado", destino)


if __name__ == "__main__":
    main()
