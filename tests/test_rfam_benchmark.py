import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "search"))

from prefilter_rfam_benchmark import first_fp_sensitivity, rfam_relevance_masks  # noqa: E402


FAMILIES = np.array(["RF1", "RF1", "RF2", "RF3", "", "RF4"], dtype=object)
CLANS = np.array(["CL1", "CL1", "CL1", "CL2", "", ""], dtype=object)


def test_family_masks_treat_unassigned_as_neutral():
    relevant, neutral = rfam_relevance_masks(0, FAMILIES, CLANS, "family")

    assert relevant.tolist() == [False, True, False, False, False, False]
    assert neutral.tolist() == [True, False, True, False, True, False]


def test_clan_masks_treat_unassigned_as_neutral_but_not_no_clan_family():
    relevant, neutral = rfam_relevance_masks(0, FAMILIES, CLANS, "clan")

    assert relevant.tolist() == [False, False, True, False, False, False]
    assert neutral.tolist() == [True, True, False, False, True, False]


def test_unassigned_target_is_skipped_before_first_false_positive():
    relevant, neutral = rfam_relevance_masks(0, FAMILIES, CLANS, "family")
    ranking = np.array([4, 1, 3], dtype=np.int64)

    assert first_fp_sensitivity(ranking, relevant, neutral) == 1.0
