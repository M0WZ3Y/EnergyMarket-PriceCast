# -*- coding: utf-8 -*-
"""Audit + fix bare Latin integers wrapped in \\lr{} in the Farsi chapters.

Convention (render-verified 2026-09-05, corrected 2026-09-06):
  * standalone integers / counts / years        -> PERSIAN digits, no \\lr
  * decimals, thousands-separated, sci-notation -> \\lr{...} with Latin digits
  * Latin identifiers / acronyms                -> \\lr{...}

A \\lr{} whose content is ONLY digits is therefore wrong and is rewritten to
Persian digits. Anything containing '.', ',', 'e-', '{,}' or a letter is left
alone. Run with --fix to apply; default is report-only.
"""
import io
import os
import re
import sys
import glob

PD = "۰۱۲۳۴۵۶۷۸۹"
EN2FA = {str(i): PD[i] for i in range(10)}

# \lr{ <digits only> }
BARE_INT = re.compile(r"\\lr\{([0-9]+)\}")


def to_persian(digits):
    return "".join(EN2FA[c] for c in digits)


def process(path, fix):
    with io.open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hits, out = [], []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("%"):
            out.append(line)
            continue
        found = BARE_INT.findall(line)
        if found:
            for d in found:
                hits.append((i, d))
            line = BARE_INT.sub(lambda m: to_persian(m.group(1)), line)
        out.append(line)
    if hits:
        print("%-26s %s" % (os.path.basename(path),
                            ", ".join("L%d:%s" % (i, d) for i, d in hits)))
        if fix:
            with io.open(path, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(out)
    return len(hits)


if __name__ == "__main__":
    fix = "--fix" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or sorted(
        glob.glob("sections/3-*.tex")
        + glob.glob("sections/4-*.tex")
        + glob.glob("sections/5-*.tex")
    )
    total = sum(process(p, fix) for p in paths)
    print("\n%d bare-Latin integer(s) %s" % (total, "fixed" if fix else "found"))
