// PROTÓTIPO (descartável) — ticket #6, candidato A: Typst.
// Os números chegam já formatados (comum.py) em sys.inputs.dados, como JSON.
#let d = json(bytes(sys.inputs.dados))

#set page(paper: "a4", margin: (x: 3cm, y: 2.5cm))
#set text(font: "Carlito", size: 11pt, lang: "pt")
#set par(spacing: 0.5em)

#let mono = "Liberation Mono"
#let f1 = $overline(f)_1$
#let titulo(n) = block(above: 1.6em, below: 0.3em, text(weight: "bold", d.titulos.at(str(n))))

// Tabela com cabeçalho colorido; `cor` em hex vindo do comum.py.
#let tabela(n, colunas, larguras, alinhamento: center, linhas) = table(
  columns: larguras,
  stroke: 0.5pt + black,
  inset: (x: 5pt, y: 2.5pt),
  align: (x, y) => if x == 0 and y > 0 { alinhamento } else { center },
  fill: (x, y) => if y == 0 { rgb(d.cores.at(str(n))) },
  table.header(..colunas.map(c => text(weight: "bold", c))),
  ..linhas.flatten(),
)

// Linhas das Tabelas 3, 5 e 6: rótulo com f̄1 e célula "média (dpa)" com o dpa menor.
#let rotulo(chave) = {
  let (antes, depois) = d.rotulos_f1.at(chave)
  if depois == none { text(size: 8.5pt, antes) } else { text(size: 8.5pt)[#antes#f1#depois] }
}
#let celula((media, dpa)) = [#media #text(size: 8.5pt)[(#dpa)]]
#let linhas_f1(t) = ("treino", "validacao", "diferenca").map(chave => (rotulo(chave), ..t.linhas.at(chave).map(celula)))

// Matriz de confusão no estilo do enunciado: fonte monoespaçada, rótulos G Y R B,
// "real" à esquerda (na linha Y) e "predito" embaixo.
#let matriz(m) = {
  set text(font: mono, size: 10pt)
  grid(
    columns: (auto, auto, 2.6em, 2.6em, 2.6em, 2.6em),
    column-gutter: 0.4em,
    row-gutter: 0.35em,
    align: (x, y) => if x >= 2 { right } else { left },
    [], [], ..d.letras.map(l => align(right, l)),
    [], d.letras.at(0), ..m.at(0),
    emph[real], d.letras.at(1), ..m.at(1),
    [], d.letras.at(2), ..m.at(2),
    [], d.letras.at(3), ..m.at(3),
    [], [], grid.cell(colspan: 4, align: center, emph[predito]),
  )
}

#titulo(1)
#tabela(1, d.t1.colunas, (6.6cm, 3.6cm), alinhamento: left, d.t1.linhas)

#titulo(2)
#tabela(2, d.t2.colunas, (4.3cm, 2.8cm, 2.8cm, 3.2cm), alinhamento: left, d.t2.linhas)

#titulo(3)
#tabela(3, d.t3.colunas, (4.3cm, 3.4cm, 4.0cm, 4.0cm), linhas_f1(d.t3))

#titulo(4)
#tabela(4, d.t4.colunas, (4.3cm, 4.0cm, 4.0cm, 4.0cm), alinhamento: left, d.t4.linhas)

#titulo(5)
#tabela(5, d.t5.colunas, (4.3cm, 3.4cm, 4.0cm, 4.0cm), linhas_f1(d.t5))

#titulo(6)
#tabela(6, d.t6.colunas, (4.3cm, 4.0cm, 3.4cm), linhas_f1(d.t6))

#titulo(7)
#tabela(7, d.t7.colunas, (3.7cm, 4.0cm, 4.0cm), d.t7.linhas.map(((r, ..v)) => (text(size: 8.5pt, r), ..v)))

#block(above: 1.6em, breakable: false)[
  #titulo(8)
  #text(weight: "bold", d.subtitulo_8)
  #table(
    columns: (7cm, 7cm),
    stroke: 0.5pt + black,
    inset: (x: 8pt, y: 4pt),
    fill: (x, y) => if y == 0 { rgb(d.cores.at("8")) },
    table.header(..("CART", "RN").map(c => align(center, text(font: mono, weight: "bold", c)))),
    matriz(d.t8.CART), matriz(d.t8.RN),
  )
]
