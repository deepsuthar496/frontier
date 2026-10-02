"""E0 KD tests: policy coefficients + alpha gating in loss_v8."""
import sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
import torch
from common import kd_alpha, KD_POLICY
from train_v8 import loss_v8


def test_policy_values():
    assert kd_alpha(True) == KD_POLICY["agree"] == 0.5
    assert kd_alpha(False) == KD_POLICY["disagree"] == 0.1
    assert kd_alpha(None) == KD_POLICY["missing"] == 0.0


def test_disagree_downweighted_not_zeroed():
    # disagree rows must receive EXACTLY the configured coefficient: prove via
    # loss difference between alpha=disagree and alpha=0 with a wrong teacher.
    torch.manual_seed(0)
    B, K = 4, 3
    stu = torch.randn(B, K)
    gold = torch.tensor([0, 1, 2, 0])
    mask = torch.ones(B, K, dtype=torch.bool)
    qt = torch.zeros(B, dtype=torch.long)
    wrong_teacher = torch.full((B, K), -1e4)
    wrong_teacher[torch.arange(B), (gold + 1) % K] = 5.0
    l0, _ = loss_v8(stu, gold, None, torch.zeros(B), mask, qt)
    lz, _ = loss_v8(stu, gold, wrong_teacher, torch.zeros(B), mask, qt)
    la, _ = loss_v8(stu, gold, wrong_teacher,
                    torch.full((B,), KD_POLICY["disagree"]), mask, qt)
    assert abs(lz.item() - l0.item()) < 1e-6, "alpha=0 must equal no-teacher loss"
    assert la.item() > lz.item() + 1e-6, "disagree KD must pull toward (wrong) teacher"


if __name__ == "__main__":
    test_policy_values()
    test_disagree_downweighted_not_zeroed()
    print("KD_TESTS_OK")
