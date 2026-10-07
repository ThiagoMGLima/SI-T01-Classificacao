// Relatório da Tarefa 1: somente as 8 tabelas da seção RELATÓRIO do enunciado, sem nenhum texto
// adicional (itens 13 e 26 de observacoes.md). Compilado por src/gerar_relatorio.py, que lê
// resultados/ e passa todos os textos das tabelas já formatados em sys.inputs.dados (JSON):
// títulos, cabeçalhos, rótulos de linha e números. Este template só faz a diagramação: cores,
// fontes, larguras e alinhamentos (itens 24 e 25).
#let d = json(bytes(sys.inputs.dados))

// Sem data de criação nos metadados: o PDF sai idêntico byte a byte a cada execução.
#set document(date: none)
#set page(paper: "a4", margin: (x: 3cm, y: 2.5cm))
// Fontes do sistema com as métricas das do enunciado: Carlito (Calibri) no texto e Liberation
// Mono (Courier New) nas matrizes de confusão.
#set text(font: "Carlito", size: 11pt, lang: "pt")
#set par(spacing: 0.5em)

#let mono = "Liberation Mono"
#let pequeno = 8.5pt

// Por tabela: cor do cabeçalho (tirada do enunciado), larguras das colunas e se os rótulos de
// linha saem pequenos e centralizados (Tabelas 3, 5, 6 e 7, como no enunciado). As tabelas
// 3 a 5 ocupam a largura do texto (15 cm); as outras são mais estreitas, como no enunciado.
#let estilo = (
  t1: (cor: rgb("#E5DFEC"), larguras: (6.6cm, 3.6cm), pequenos: false),
  t2: (cor: rgb("#DAEEF3"), larguras: (4.3cm, 2.9cm, 2.9cm, 2.9cm), pequenos: false),
  t3: (cor: rgb("#DAEEF3"), larguras: (4cm, 1fr, 1fr, 1fr), pequenos: true),
  t4: (cor: rgb("#FDE9D9"), larguras: (4cm, 1fr, 1fr, 1fr), pequenos: false),
  t5: (cor: rgb("#FDE9D9"), larguras: (4cm, 1fr, 1fr, 1fr), pequenos: true),
  t6: (cor: rgb("#B2A1C7"), larguras: (4cm, 3.7cm, 3.7cm), pequenos: true),
  t7: (cor: rgb("#B2A1C7"), larguras: (4cm, 3.7cm, 3.7cm), pequenos: true),
  t8: (cor: rgb("#B2A1C7"), larguras: (7cm, 7cm)),
)

#let titulo(texto) = block(above: 1.6em, below: 0.3em, text(weight: "bold", texto))

// O "f̄1" (f com mácron) dos rótulos das Tabelas 3, 5 e 6 sai na fonte matemática do Typst.
#let f1 = $overline(f)_1$
#let rotulo(r) = r.split("f\u{0304}1").join(f1)

// Célula de texto, ou [média, DPA] das Tabelas 3, 5 e 6: "0.88212 (0.01074)", com o DPA menor.
#let celula(c) = if type(c) == array {
  let (media, dpa) = c
  [#media #text(size: pequeno)[(#dpa)]]
} else { c }

#let tabela(t, e) = table(
  columns: e.larguras,
  stroke: 0.5pt + black,
  inset: (x: 5pt, y: 2.5pt),
  align: (x, y) => if x == 0 and y > 0 and not e.pequenos { left } else { center },
  fill: (x, y) => if y == 0 { e.cor },
  table.header(..t.colunas.map(c => text(weight: "bold", c))),
  ..t.linhas.map(((r, ..celulas)) => (
    if e.pequenos { text(size: pequeno, rotulo(r)) } else { rotulo(r) },
    ..celulas.map(celula),
  )).flatten(),
)

// Matriz de confusão no estilo do enunciado: fonte monoespaçada, letras das classes de triagem,
// classe real nas linhas (rótulo na linha da segunda classe) e classe predita nas colunas.
#let matriz(t, m) = {
  set text(font: mono, size: 10pt)
  let l = t.letras
  grid(
    columns: (auto, auto, 2.6em, 2.6em, 2.6em, 2.6em),
    column-gutter: 0.4em,
    row-gutter: 0.35em,
    align: (x, y) => if x >= 2 { right } else { left },
    [], [], ..l.map(letra => align(right, letra)),
    [], l.at(0), ..m.at(0),
    emph(t.real), l.at(1), ..m.at(1),
    [], l.at(2), ..m.at(2),
    [], l.at(3), ..m.at(3),
    [], [], grid.cell(colspan: 4, align: center, emph(t.predito)),
  )
}

#for n in range(1, 8) {
  let chave = "t" + str(n)
  titulo(d.at(chave).titulo)
  tabela(d.at(chave), estilo.at(chave))
}

#block(above: 1.6em, breakable: false)[
  #titulo(d.t8.titulo)
  #text(weight: "bold", d.t8.subtitulo)
  #table(
    columns: estilo.t8.larguras,
    stroke: 0.5pt + black,
    inset: (x: 8pt, y: 4pt),
    fill: (x, y) => if y == 0 { estilo.t8.cor },
    table.header(..d.t8.colunas.map(c => align(center, text(font: mono, weight: "bold", c)))),
    ..d.t8.matrizes.map(m => matriz(d.t8, m)),
  )
]
