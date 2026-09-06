# -*- coding: utf-8 -*-
"""Report doubled reference words and any surviving hardcoded refs.

cleveref already prints "بخش"/"شکل"/"جدول" itself, so a literal word left in
front of a \\cref produces "جدول بخش ۴-۲". This finds those, plus any Persian
"بخش ۳-۵"-style reference the conversion missed.
"""
import io
import re
import sys
import glob

DOUBLED = re.compile("(جدول|شکل|بخش)\\s*\\\\cref\\{([^}]+)\\}")
LEFTOVER = re.compile("(بخش|شکل|جدول)\\s+[۰-۹]+(?:[-‐‑][۰-۹]+)+")

sys.stdout.reconfigure(encoding="utf-8")

doubled = leftover = 0
for f in sorted(glob.glob("sections/[345]-*.tex")):
    for i, line in enumerate(io.open(f, encoding="utf-8"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in DOUBLED.finditer(line):
            print("DOUBLED  %s:%d  %s" % (f, i, m.group(0)))
            doubled += 1
        for m in LEFTOVER.finditer(line):
            print("LEFTOVER %s:%d  %s" % (f, i, m.group(0)))
            leftover += 1

print("\n%d doubled, %d leftover hardcoded refs" % (doubled, leftover))
