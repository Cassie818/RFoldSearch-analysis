"""Generate S4 with the clean accuracy/selectivity supplement style."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("plot_supp_codebook_and_kmer.py")),
               run_name="__main__")
