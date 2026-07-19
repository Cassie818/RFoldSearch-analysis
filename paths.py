"""Central path resolution.

The analysis scripts read large inputs (tokenized database, descriptor tensors,
US-align / RMalign caches) that are NOT bundled in this repository because of
their size. Point DATA_ROOT at the directory holding `files/` and `Splits/`.

    export RFOLDSEEK_DATA=/path/to/RFoldSeek

Figures are written to `figures/` by default; override with RFOLDSEEK_FIGURES.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get("RFOLDSEEK_DATA", _HERE)
FIG_DIR = os.environ.get("RFOLDSEEK_FIGURES", os.path.join(_HERE, "figures"))
RESULTS_DIR = os.path.join(_HERE, "results")

SRC_MODELS = os.path.join(_HERE, "src", "models")
SRC_SEARCH = os.path.join(_HERE, "src", "search")

os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.isdir(os.path.join(DATA_ROOT, "files")):
    import warnings
    warnings.warn(
        f"DATA_ROOT={DATA_ROOT} has no files/ directory. "
        "Set RFOLDSEEK_DATA to the directory containing files/ and Splits/. "
        "Figure scripts that read from results/ will still work.",
        RuntimeWarning,
    )
