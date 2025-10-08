import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BAD = []
pat = re.compile(r"st\.(markdown|write|text)\(\s*[fr]?[\"']\s*</div>\s*[\"']", re.I)
for p in ROOT.rglob("*.py"):
    if "venv" in p.parts:
        continue
    s = p.read_text(encoding="utf-8", errors="ignore")
    if pat.search(s):
        BAD.append(str(p))
if BAD:
    print("[ERROR] Cierres </div> huérfanos en:", *BAD, sep="\n - ")
    sys.exit(1)
print("[OK] Sin cierres </div> huérfanos detectados.")
