# -*- coding: utf-8 -*-
"""Enforce the render-verified numeral/typography convention on chapter 3-5 files.

Verified 2026-09-05 by compiling numtest.tex with the real B Nazanin font and
reading the rasterized page (see WRITING_HANDOFF A1, which asks for exactly
this check):

  * Persian decimal mark U+066B and Persian thousands mark U+066C are NOT in
    B Nazanin. They print as tofu AND the number reorders RTL.
  * A Persian slash decimal (3/557) renders but REORDERS to 557/3.
  * Latin digits inside \lr{} render correctly in both cases.
  * Hamza-above U+0654 (as in "دورهٔ") is NOT in B Nazanin -> tofu.
    The approved ch1-2 files contain zero U+0654; they use "ه‌ی".
  * Plain Persian-digit integers render correctly and are left alone.

So: integers -> Persian digits; decimals and thousands -> \lr{latin}.
"""
import re
import sys
import io
import glob
import os

PD = "۰۱۲۳۴۵۶۷۸۹"
FA2EN = {ord(c): str(i) for i, c in enumerate(PD)}
DEC = "٫"   # Persian decimal separator
THO = "٬"   # Persian thousands separator
ZWNJ = "‌"

# A Persian-digit run that contains a decimal or thousands mark.
NUM = re.compile("[" + PD + "]+(?:[" + DEC + THO + "][" + PD + "]+)+")


def convert_number(m):
    s = m.group(0)
    latin = s.translate(FA2EN)
    latin = latin.replace(DEC, ".").replace(THO, "{,}")
    return "\\lr{" + latin + "}"


def fix(text):
    # 1. hamza-above ezafe -> ZWNJ + yeh  (دورهٔ -> دوره‌ی)
    text = text.replace("هٔ", "ه" + ZWNJ + "ی")
    # 2. any stray hamza-above left on another letter: drop it rather than
    #    print tofu. None are expected; report if found.
    stray = text.count("ٔ")
    # 3. Persian decimals / thousands -> \lr{...}
    text = NUM.sub(convert_number, text)
    return text, stray


def main(paths):
    for p in paths:
        with io.open(p, encoding="utf-8") as f:
            src = f.read()
        out, stray = fix(src)
        if out != src:
            with io.open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(out)
            print("fixed  %-26s (stray hamza left: %d)" % (os.path.basename(p), stray))
        else:
            print("clean  %-26s" % os.path.basename(p))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("sections/3-*.tex") +
                      glob.glob("sections/4-*.tex") +
                      glob.glob("sections/5-*.tex"))
    main(args)
