import os
import ast
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    "os", "sys", "time", "json", "re", "random", "math", "argparse",
    "typing", "pathlib", "collections", "itertools", "functools", "copy",
    "shutil", "glob", "hashlib", "subprocess", "threading", "datetime",
    "io", "struct", "dataclasses", "unittest", "logging", "gc"
}

external_pkgs = set()
local_modules = {"core", "dataset_cache", "models"}

for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as src:
                    tree = ast.parse(src.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            pkg = n.name.split(".")[0]
                            if pkg not in stdlib and pkg not in local_modules:
                                external_pkgs.add(pkg)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            pkg = node.module.split(".")[0]
                            if pkg not in stdlib and pkg not in local_modules:
                                external_pkgs.add(pkg)
            except Exception:
                pass

print("=" * 60)
print("📦 【Studio 35 検出外部インポート一覧】")
print("=" * 60)
for p in sorted(list(external_pkgs)):
    try:
        mod = __import__(p)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {p:<15} (version: {ver})")
    except ImportError:
        print(f"  ❌ {p:<15} (未インストール)")
print("=" * 60)
