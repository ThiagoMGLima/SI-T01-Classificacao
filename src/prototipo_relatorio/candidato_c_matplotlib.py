"""PROTÓTIPO (descartável) — ticket #6, candidato C: matplotlib (PdfPages), sem dependência nova.

Desenha cada tabela "à mão" numa figura A4: retângulos e textos posicionados em cm a partir
do canto superior esquerdo. f̄1 sai pelo mathtext do matplotlib ($\\overline{f}_1$). Larguras de
coluna e quebra de página são fixas no código (o matplotlib não diagrama texto).

Uso: uv run python src/prototipo_relatorio/candidato_c_matplotlib.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from comum import COR, LETRAS, ROTULOS_F1, SAIDA, SUBTITULO_8, TITULOS, montar_relatorio

plt.rcParams.update({
    "font.family": "Carlito",
    "font.size": 11,
    "mathtext.fontset": "custom",
    "mathtext.it": "Caladea:italic",
    "mathtext.rm": "Caladea",
    "mathtext.cal": "Caladea:italic",  # evita aviso de fonte "cursive" ausente
    "pdf.fonttype": 42,  # TrueType embutida (texto selecionável)
})
LARG, ALT = 21.0, 29.7  # A4 em cm
MARGEM_X, MARGEM_Y = 3.0, 2.5
H_LINHA = 0.48
F1 = r"$\,\overline{f}_1$"
MONO = "Liberation Mono"


class Pagina:
    """Uma figura A4 com cursor vertical (em cm, a partir do topo)."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.nova()

    def nova(self):
        self.fig = plt.figure(figsize=(LARG / 2.54, ALT / 2.54))
        self.y = MARGEM_Y

    def fechar(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def texto(self, x, y, s, **kw):
        return self.fig.text(x / LARG, 1 - y / ALT, s, **kw)

    def retangulo(self, x, y, w, h, cor=None):
        self.fig.patches.append(Rectangle(
            (x / LARG, 1 - (y + h) / ALT), w / LARG, h / ALT, transform=self.fig.transFigure,
            facecolor=cor or "none", edgecolor="black", linewidth=0.6))

    def largura(self, s, **kw):
        """Largura do texto em cm (para centralizar "média (dpa)" com tamanhos de fonte diferentes)."""
        t = self.texto(0, 0, s, **kw)
        bbox = t.get_window_extent(self.fig.canvas.get_renderer())
        t.remove()
        return bbox.width / self.fig.dpi * 2.54

    def titulo(self, s, espaco=0.75):
        self.y += espaco
        self.texto(MARGEM_X, self.y, s, weight="bold", va="baseline")
        self.y += 0.15

    def tabela(self, n, colunas, linhas, larguras, esq=True, rotulo_pequeno=False):
        """Células: str, ou tupla (média, dpa) para "média (dpa)" com o dpa menor."""
        for i, linha in enumerate([colunas, *linhas]):
            x = MARGEM_X
            for j, (celula, w) in enumerate(zip(linha, larguras)):
                self.retangulo(x, self.y, w, H_LINHA, COR[n] if i == 0 else None)
                yb = self.y + H_LINHA / 2
                kw = {"va": "center"}
                if i == 0:
                    self.texto(x + w / 2, yb, celula, ha="center", weight="bold", **kw)
                elif isinstance(celula, tuple):
                    media, dpa = celula[0] + " ", f"({celula[1]})"
                    w1, w2 = self.largura(media), self.largura(dpa, size=8.5)
                    x0 = x + (w - w1 - w2) / 2
                    # mesma linha de base para os dois tamanhos de fonte
                    self.texto(x0, yb + 0.13, media, va="baseline")
                    self.texto(x0 + w1, yb + 0.13, dpa, size=8.5, va="baseline")
                elif j == 0 and rotulo_pequeno:
                    self.texto(x + w / 2, yb, celula, ha="center", size=8.5, **kw)
                elif j == 0 and esq:
                    self.texto(x + 0.15, yb, celula, **kw)
                else:
                    self.texto(x + w / 2, yb, celula, ha="center", **kw)
                x += w
            self.y += H_LINHA

    def matrizes(self, m_cart, m_rn):
        w, h_cab, h_corpo = 7.0, H_LINHA, 6 * 0.45 + 0.3
        for k, (nome, m) in enumerate([("CART", m_cart), ("RN", m_rn)]):
            x = MARGEM_X + k * w
            self.retangulo(x, self.y, w, h_cab, COR[8])
            self.texto(x + w / 2, self.y + h_cab / 2, nome, ha="center", va="center", family=MONO, weight="bold")
            self.retangulo(x, self.y + h_cab, w, h_corpo)
            # grade: "real" | letra da linha | 4 colunas de números alinhados à direita
            x_real, x_letra, x_cols = x + 0.3, x + 1.4, [x + 2.6 + c * 1.0 for c in range(4)]
            y0 = self.y + h_cab + 0.35
            for c, l in enumerate(LETRAS):
                self.texto(x_cols[c], y0, l, ha="right", va="center", family=MONO, size=10)
            for r, l in enumerate(LETRAS):
                yr = y0 + (r + 1) * 0.45
                if r == 1:
                    self.texto(x_real, yr, "real", va="center", family=MONO, style="italic", size=10)
                self.texto(x_letra, yr, l, va="center", family=MONO, size=10)
                for c in range(4):
                    self.texto(x_cols[c], yr, m[r][c], ha="right", va="center", family=MONO, size=10)
            self.texto((x_cols[0] + x_cols[3]) / 2 - 0.2, y0 + 5 * 0.45, "predito", ha="center", va="center",
                       family=MONO, style="italic", size=10)
        self.y += h_cab + h_corpo


def linhas_f1(t):
    linhas = []
    for chave, (antes, depois) in ROTULOS_F1.items():
        rotulo = antes if depois is None else f"{antes}{F1}{depois}"
        linhas.append([rotulo, *t["linhas"][chave]])
    return linhas


def main():
    r = montar_relatorio()
    SAIDA.mkdir(exist_ok=True)
    destino = SAIDA / "candidato_c_matplotlib.pdf"
    larg_res = [4.3, 3.4, 4.0, 4.0]
    with PdfPages(destino) as pdf:
        p = Pagina(pdf)
        p.titulo(TITULOS[1], espaco=0)
        p.tabela(1, r[1]["colunas"], r[1]["linhas"], [6.6, 3.6])
        p.titulo(TITULOS[2])
        p.tabela(2, r[2]["colunas"], r[2]["linhas"], [4.3, 2.8, 2.8, 3.2])
        p.titulo(TITULOS[3])
        p.tabela(3, r[3]["colunas"], linhas_f1(r[3]), larg_res, rotulo_pequeno=True)
        p.titulo(TITULOS[4])
        p.tabela(4, r[4]["colunas"], r[4]["linhas"], [4.3, 4.0, 4.0, 4.0])
        p.titulo(TITULOS[5])
        p.tabela(5, r[5]["colunas"], linhas_f1(r[5]), larg_res, rotulo_pequeno=True)
        p.titulo(TITULOS[6])
        p.tabela(6, r[6]["colunas"], linhas_f1(r[6]), [4.3, 4.0, 3.4], rotulo_pequeno=True)
        p.titulo(TITULOS[7])
        p.tabela(7, r[7]["colunas"], r[7]["linhas"], [3.7, 4.0, 4.0], rotulo_pequeno=True)
        # A quebra de página é manual: a Tabela 8 não cabe no resto da página 1? Então página 2.
        if p.y + 5.5 > ALT - MARGEM_Y:
            p.fechar()
            p.nova()
            p.titulo(TITULOS[8], espaco=0)
        else:
            p.titulo(TITULOS[8])
        p.y += 0.35
        p.texto(MARGEM_X, p.y, SUBTITULO_8, weight="bold", va="baseline")
        p.y += 0.15
        p.matrizes(r[8]["CART"], r[8]["RN"])
        p.fechar()
    print("gravado", destino)


if __name__ == "__main__":
    main()
