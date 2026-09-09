FINAL THESIS DELIVERY
=====================
Built 2026-09-09 from a clean tree (all .aux/.toc/.lof/.lot/.bbl wiped,
then xelatex -> bibtex -> xelatex x3).

thesis-final.pdf   131 pages. Complete document, in order:
                     cover / Farsi title page          1
                     defense form + تعهدنامه اصالت اثر  3-5
                     تقدیم (dedication)                 6
                     تقدیر و تشکر (acknowledgments)     7
                     چکیده + کلیدواژه‌ها                8
                     فهرست مطالب / تصاویر / جداول       ا-ز
                     فهرست نمادها                      ط
                     Chapter 1  مقدمه                   printed 1
                     Chapter 2  پیشینه                  printed 10
                     Chapter 3  روش تحقیق               printed 21
                     Chapter 4  نتایج و تحلیل           printed 59
                     Chapter 5  جمع‌بندی                printed 88
                     کتاب‌نامه (42 refs)                printed 101
                     پیوست                             printed 107
                     واژه‌نامه فا->en                   printed 108
                     واژه‌نامه en->فا                   printed 111
                     English abstract + English title page

                   Body is printed pages 1-100, exactly the 100pp target.

tex/               Complete, self-contained sources.
                   VERIFIED: unpacked into an empty directory and built with
                     xelatex -> bibtex -> xelatex -> xelatex
                   it produces the identical 131-page PDF, 0 errors,
                   0 undefined references. Nothing outside this folder is
                   needed -- logos and all 17 figures are included, and the
                   flat layout is what \graphicspath{{./}...} already expects.

  AUTthesis.tex      master file -- build this one
  AUTthesis.cls      Amirkabir class (unmodified)
  commands.tex       preamble
  chapter1-5.tex     chapter spines (\input only, no prose)
  sections/*.tex     all prose, one file per section (42 files)
  fa_title.tex       Farsi title page + abstract + keywords
  en_title.tex       English title page
  taid.tex           defense form table + تعهدنامه اصالت اثر
  Chant.tex          تقدیم
  acknowledgement.tex تقدیر و تشکر, then prints the Farsi abstract
  en-abstract.tex    English abstract + keywords
  list-of-symbols.tex فهرست نمادها
  dicfa2en.tex       واژه‌نامه فارسی به انگلیسی
  dicen2fa.tex       واژه‌نامه انگلیسی به فارسی
  TOC-TOF-LOT.tex    the three auto-generated lists
  references.bib     bibliography source
  AUTthesis.bbl      generated bibliography (unsrt-fa: ordered by first
                     citation, never renumber by hand)
  *.png, besm.jpg    logos and the 17 figures

BUILD VERIFICATION (clean build, 2026-09-09)
  0 errors
  0 undefined references
  0 undefined citations
  17/17 figures resolve      (LOF lists 17)
  15 tables                  (LOT lists 15)
  42 bibliography entries, all cited keys present
  12 "Missing character" warnings -- see note below

THE 12 MISSING-CHARACTER WARNINGS ARE HARMLESS AND NOT FIXABLE HERE.
They are not content. bidi's own multicol patch
(multicol-xetex-bidi.def lines 61 and 65) measures the descender depth of a
lowercase Latin "p" to align column bottoms:
    \setbox\z@\hbox{p}\global\dimen\tw@\dp\z@
    \rlap{\phantom p}%
Inside the Persian glossary the current font is B Nazanin, which has no Latin
"p", so every shipped glossary page logs exactly two. Both are invisible by
construction: the \setbox is measured and never shipped, and \phantom prints
nothing. Count scales with glossary page count (8 over the four placeholder
pages, 12 over the six real ones), not with what the glossary says.

TWO FIELDS STILL NEED THE AUTHOR
  1. Surname. Set to the full registrar form «عظیم‌پور چرندابی» per the
     approved proposal. If the registrar record omits «چرندابی», edit
     \surname in tex/fa_title.tex -- the declaration page and the
     signature block both read from it. English twin: \latinsurname in
     tex/en_title.tex.
  2. Defense date. Currently the placeholder «[ماه و سال دفاع]» in
     \thesisdate (tex/fa_title.tex) and "[Month & Year of Defense]"
     in \latinthesisdate (tex/en_title.tex).

OPTIONAL, COSMETIC
  - The چکیده page prints "1" because the template restarts abjad numbering
    after it. Stock AUTthesis behaviour; moving \pagenumbering{alph} one line
    earlier in AUTthesis.tex fixes it if the department objects.
  - \emph{} renders unstyled: B Nazanin has no italic face. Adding
    \renewcommand{\emph}[1]{\textbf{#1}} to commands.tex would make the
    emphasis visible. Left as-is because it is consistent across all five
    chapters.
  - The abstract names لاگو و همکاران in prose rather than carrying \cite{},
    because the AUT template instructs that abstracts not cite references.
