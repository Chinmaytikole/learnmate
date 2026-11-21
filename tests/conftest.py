# conftest.py
import os
import pathlib

# Ensure matplotlib uses a writable workspace directory in sandbox on Windows
mpl_dir = pathlib.Path(__file__).resolve().parent.parent / ".mpl_cache"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")

try:
    import faiss
except Exception:
    pass
