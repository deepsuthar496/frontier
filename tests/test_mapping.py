"""E0 mapping tests: canonical/model coordinates, round-trip, typed semantics."""
import itertools
import sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
from common import (canonical_options, canonical_gold_idx, model_order,
                    gold_model_idx, decode_pred)


def test_choice_roundtrip_all_perms():
    keys = ["a", "b", "c"]
    q = {"t": "choice", "ins": "i", "crit": {k: f"d{k}" for k in keys}}
    for perm in itertools.permutations(range(3)):
        for gold in keys:
            ci = canonical_gold_idx(q, gold)
            mi = gold_model_idx(q, gold, list(perm))
            assert canonical_options(q)[ci] == gold
            assert model_order(q, list(perm))[mi] == ci
            assert decode_pred(mi, q, list(perm)) == gold


def test_score_semantics():
    q = {"t": "score", "ins": "i", "crit": ["lo", "mid", "hi"]}
    assert canonical_options(q) == ["0", "1", "2"]
    assert canonical_gold_idx(q, 2) == 2
    assert decode_pred(0, q) == 0
    assert isinstance(decode_pred(1, q), int)


def test_noul_semantics():
    q = {"t": "noul", "ins": "i", "crit": {}}
    assert canonical_options(q) == ["false", "true"]
    assert canonical_gold_idx(q, "yes") == 1
    assert canonical_gold_idx(q, "no") == 0
    assert canonical_gold_idx(q, True) == 1
    assert decode_pred(1, q) == 1


def test_labels_vs_criteria_order():
    # audit case: labels display order differs from criteria order; mapping
    # must still resolve the expected VALUE, not a positional coincidence.
    crit = {"spreadsheet_formula": "x", "office_file_repair": "y",
            "data_pipeline": "z", "desktop_support": "w", "finance_review": "v"}
    q = {"t": "choice", "ins": "i", "crit": crit}
    assert decode_pred(gold_model_idx(q, "office_file_repair"), q) == "office_file_repair"


if __name__ == "__main__":
    test_choice_roundtrip_all_perms()
    test_score_semantics()
    test_noul_semantics()
    test_labels_vs_criteria_order()
    print("MAPPING_TESTS_OK")
