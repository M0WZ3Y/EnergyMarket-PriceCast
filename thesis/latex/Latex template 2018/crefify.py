# -*- coding: utf-8 -*-
"""Convert hardcoded Persian cross-references into \\cref, and add the labels.

cleveref is loaded in commands.tex with Persian names already configured
(\\crefname{section}{بخش}, {figure}{شکل}, {table}{جدول}, {equation}{برابری}),
so \\cref{sec:3-5} renders exactly the string "بخش ۳-۵" that the prose
currently hardcodes. The conversion is therefore visually neutral and makes
the references live.

Labelling scheme (stable across chapters, so ch4/ch5 forward refs resolve
once those chapters carry their labels):
    \\section{...}    -> \\label{sec:<n>-<m>}
    \\subsection{...} -> \\label{sec:<n>-<m>-<k>}
Starred headings (\\subsection*) are unnumbered and are never labelled or
referenced.

Run with --fix to apply; default is report-only.
"""
import io
import os
import re
import sys
import glob

PD = "۰۱۲۳۴۵۶۷۸۹"
FA2EN = {PD[i]: str(i) for i in range(10)}

# "بخش ۳-۵" / "بخش ۳-۷-۱"  (Persian digits joined by ASCII or Persian hyphen)
SECREF = re.compile(r"بخش\s+([۰-۹]+(?:[-‐‑][۰-۹]+)+)")
# "شکل ۳-۱"
FIGREF = re.compile(r"شکل\s+([۰-۹]+(?:[-‐‑][۰-۹]+)+)")
# "جدول \ref{...}"  -> \cref{...}
TABREF = re.compile(r"جدول\s+\\ref\{([^}]+)\}")

# figure number (as printed) -> label, for chapter 3
FIGMAP = {
    "3-1": "fig:price-dist",
    "3-2": "fig:hourly-seasonality",
    "3-3": "fig:weekly-seasonality",
    "3-4": "fig:annual-seasonality",
    "3-5": "fig:vol-clustering",
    "3-6": "fig:acf-pacf",
    "3-7": "fig:exog-corr",
    "3-8": "fig:structural-breaks",
    "3-9": "fig:daily-baseload",
}


def fa2en(s):
    return "".join(FA2EN.get(c, c) for c in s)


def norm(num):
    """۳-۷-۱ -> 3-7-1"""
    return re.sub(r"[‐‑]", "-", fa2en(num))


def add_labels(text, stem):
    """Put \\label{sec:...} after the numbered heading this file owns."""
    # stem like '3-3-1-benchmark' -> section id '3-3-1'
    m = re.match(r"(\d+(?:-\d+)*)", stem)
    if not m:
        return text, None
    sec_id = m.group(1)
    label = "sec:" + sec_id
    if "\\label{" + label + "}" in text:
        return text, label
    # attach to the first numbered \section{...} or \subsection{...}
    pat = re.compile(r"(\\(?:sub)?section\{(?:[^{}]|\{[^{}]*\})*\})")
    m2 = pat.search(text)
    if not m2:
        return text, None
    ins = m2.end()
    return text[:ins] + "\n\\label{" + label + "}" + text[ins:], label


def convert(text):
    n = {"sec": 0, "fig": 0, "tab": 0}

    def sec_sub(m):
        n["sec"] += 1
        return "\\cref{sec:" + norm(m.group(1)) + "}"

    def fig_sub(m):
        key = norm(m.group(1))
        if key not in FIGMAP:
            return m.group(0)
        n["fig"] += 1
        return "\\cref{" + FIGMAP[key] + "}"

    def tab_sub(m):
        n["tab"] += 1
        return "\\cref{" + m.group(1) + "}"

    text = TABREF.sub(tab_sub, text)
    text = SECREF.sub(sec_sub, text)
    text = FIGREF.sub(fig_sub, text)
    return text, n


def main():
    fix = "--fix" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or sorted(
        glob.glob("sections/3-*.tex")
        + glob.glob("sections/4-*.tex")
        + glob.glob("sections/5-*.tex")
    )
    tot = {"sec": 0, "fig": 0, "tab": 0}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        src = io.open(p, encoding="utf-8").read()
        out, label = add_labels(src, stem)
        out, n = convert(out)
        for k in tot:
            tot[k] += n[k]
        if any(n.values()) or out != src:
            print("%-26s label=%-10s sec=%d fig=%d tab=%d"
                  % (os.path.basename(p), label or "-", n["sec"], n["fig"], n["tab"]))
            if fix:
                io.open(p, "w", encoding="utf-8", newline="\n").write(out)
    print("\ntotal: %d section refs, %d figure refs, %d table refs %s"
          % (tot["sec"], tot["fig"], tot["tab"], "converted" if fix else "found"))


if __name__ == "__main__":
    main()
