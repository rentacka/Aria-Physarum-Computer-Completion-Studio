import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = r"E:\学習用データセット"
for root, dirs, files in os.walk(base):
    for f in files:
        fl = f.lower()
        if any(k in fl for k in ['think', 'amenokaku', 'aozora', 'onomatopoeia', 'alpaca', 'persona']):
            print(os.path.join(root, f))
