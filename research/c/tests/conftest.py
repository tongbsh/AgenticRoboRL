import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
REPO = C_DIR.parents[1]
for p in (HERE, C_DIR, REPO / "rl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
RPENT_DIR = os.environ.get("RPENT_DIR")
if RPENT_DIR and RPENT_DIR not in sys.path:
    sys.path.insert(0, RPENT_DIR)
