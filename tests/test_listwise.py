"""F2 tests: legacy equivalence (flag off) + listwise forward/grad (flag on)."""
import sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
import torch
from transformers import AutoConfig, AutoModel
from model import FrontierDecisionEngine


def tiny_enc():
    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(
        "/teamspace/studios/this_studio/frontier/frontier_ckpt_v3/encoder")
    cfg.num_hidden_layers = 2
    return AutoModel.from_config(cfg, attn_implementation="sdpa")


def test_legacy_equivalence_and_ckpt():
    # old checkpoint must load STRICT into listwise=0 model (no new params)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer  # noqa
    enc = AutoModel.from_config(
        AutoConfig.from_pretrained(
            "/teamspace/studios/this_studio/frontier/frontier_ckpt_v7/encoder"),
        attn_implementation="sdpa")
    m = FrontierDecisionEngine(enc, head_layers=2)  # listwise default 0
    m.load_state_dict(
        load_file("/teamspace/studios/this_studio/frontier/frontier_ckpt_v7/model.safetensors"),
        strict=True)
    assert not any("listwise" in k for k in m.state_dict())
    print("LEGACY_CKPT_OK")


def test_listwise_forward_grad():
    torch.manual_seed(0)
    enc = tiny_enc()
    m = FrontierDecisionEngine(enc, head_layers=1, listwise_layers=2)
    m.train()
    B, L, K = 2, 24, 3
    ids = torch.randint(0, 500, (B, L))
    att = torch.ones(B, L, dtype=torch.long)
    mpos = torch.tensor([[2, 5, 8], [3, 6, 9]])
    mmask = torch.ones(B, K, dtype=torch.bool)
    qt = torch.tensor([0, 1])
    out = m(ids, att, mpos, mmask, qt)
    assert out["choice_logits"].shape == (B, K)
    loss = out["choice_logits"].sum() + out["score"].sum()
    loss.backward()
    assert m.listwise.layers[0].self_attn.in_proj_weight.grad is not None
    assert torch.isfinite(out["choice_logits"]).all()
    print("LISTWISE_OK")


def test_recurrent_identity_and_depth():
    torch.manual_seed(0)
    enc = tiny_enc()
    m0 = FrontierDecisionEngine(enc, head_layers=1, rec_steps=0)
    m1 = FrontierDecisionEngine(enc, head_layers=1, rec_steps=3)
    m1.load_state_dict(m0.state_dict(), strict=True)  # zero new params
    m0.eval()
    m1.eval()
    B, L, K = 2, 24, 3
    ids = torch.randint(0, 500, (B, L))
    att = torch.ones(B, L, dtype=torch.long)
    mpos = torch.tensor([[2, 5, 8], [3, 6, 9]])
    mmask = torch.ones(B, K, dtype=torch.bool)
    qt = torch.tensor([0, 2])
    with torch.no_grad():
        o0 = m0(ids, att, mpos, mmask, qt)
        o1 = m1(ids, att, mpos, mmask, qt)
    assert not torch.equal(o0["choice_logits"], o1["choice_logits"])
    # rec>0 must actually change the representation (loop runs)
    assert len(o1["all_h"]) == 4 and len(o0["all_h"]) == 1
    assert not torch.equal(o1["all_h"][0], o1["all_h"][-1])
    assert torch.isfinite(o1["choice_logits"]).all()
    print("RECURRENT_OK")


if __name__ == "__main__":
    test_legacy_equivalence_and_ckpt()
    test_listwise_forward_grad()
    test_recurrent_identity_and_depth()
