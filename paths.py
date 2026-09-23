"""Central path resolution.

Two kinds of input are read by the scripts here:

* Bundled results (`files/ablation/*.json`, `files/production/retrieval/*.json`)
  ship with this repository, so every figure can be redrawn after a plain clone.
* Large artifacts (tokenized database, descriptor tensors, trained model,
  US-align / RMalign caches) are NOT bundled. Point DATA_ROOT at the directory
  holding `files/` and `Splits/` to regenerate results from scratch:

      export RFOLDSEEK_DATA=/path/to/RFoldSeek

Bundled results take precedence: setting RFOLDSEEK_DATA lets the heavy scripts
find their inputs without silently swapping the published numbers for whatever
happens to sit in the external tree. Set RFOLDSEEK_PREFER_EXTERNAL=1 to reverse
that and read results from DATA_ROOT instead.

Figures are written to `figures/`; override with RFOLDSEEK_FIGURES.
"""
import os
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get("RFOLDSEEK_DATA", _HERE)
FIG_DIR = os.environ.get("RFOLDSEEK_FIGURES", os.path.join(_HERE, "figures"))
PREFER_EXTERNAL = os.environ.get("RFOLDSEEK_PREFER_EXTERNAL", "") == "1"

SRC_MODELS = os.path.join(_HERE, "src", "models")
SRC_SEARCH = os.path.join(_HERE, "src", "search")

# --- External inputs (not bundled) -------------------------------------------
# Needed only to regenerate results from raw structures; every figure can be
# redrawn from the bundled results without them. Each is an environment
# variable; the defaults are the locations the manuscript was produced with.
_PARENT = os.path.dirname(_HERE)


def _first_existing(env, *candidates):
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# Released RFoldseek tool (search CLI, model assets, RiboSeek wrappers).
RFOLDSEEK_REPO = _first_existing("RFOLDSEEK_REPO", os.path.join(_PARENT, "RFoldseek-upstream"))
# Experimental RNAsolo structures (mmCIF): Fig 2 Infernal queries, Fig 4 experimental queries.
NR80_DIR = _first_existing(
    "RFOLDSEEK_NR80", os.path.join(_PARENT, "NR80"),
    os.path.expanduser("~/Library/CloudStorage/OneDrive-TheUniversityofQueensland/RFoldseek/NR80"))
# NuFold predicted-structure release (structures/rank1, metadata/data.csv): Fig 4.
NUFOLD_ROOT = _first_existing(
    "RFOLDSEEK_NUFOLD_ROOT", os.path.join(_PARENT, "RFoldSeek", "data", "predicted", "nufold_9_14_2026"))
# Rfam 3D seed structures for the token-alignment analysis (alignment, cases).
RFAM3D_STRUCTURES = _first_existing(
    "RFOLDSEEK_RFAM3D_STRUCTURES",
    os.path.join(_PARENT, "RFoldseek-structures", "files", "ablation", "rfam3d_reference", "structures"))
# Third-party binaries. An empty directory means "take the executable from PATH".
USALIGN = _first_existing("RFOLDSEEK_USALIGN", os.path.join(_PARENT, "RFoldSeek", ".tools", "USalign"))
BLAST_BIN = os.environ.get("RFOLDSEEK_BLAST_BIN", "")
INFERNAL_BIN = os.environ.get("RFOLDSEEK_INFERNAL_BIN", "")

os.makedirs(FIG_DIR, exist_ok=True)


def resolve(relpath):
    """Absolute path for a repo-relative input.

    Prefers the bundled copy so published figures stay reproducible even when
    DATA_ROOT points at a working tree whose results are mid-regeneration.
    """
    bundled = os.path.join(_HERE, relpath)
    external = os.path.join(DATA_ROOT, relpath)
    if PREFER_EXTERNAL:
        return external if os.path.exists(external) else bundled
    return bundled if os.path.exists(bundled) else external


if DATA_ROOT != _HERE and not os.path.isdir(os.path.join(DATA_ROOT, "files")):
    warnings.warn(
        f"RFOLDSEEK_DATA={DATA_ROOT} has no files/ directory; "
        "scripts needing the large artifacts will fail.",
        RuntimeWarning,
    )
