"""Export the champion checkpoint to ONNX (opset 17, dynamic batch/length/options).

Usage: python3 export_onnx.py [--ckpt frontier_ckpt_v3]
Verifies the export with onnxruntime (CPU).
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import FrontierDecisionEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="frontier_ckpt_v3")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file
    enc = AutoModel.from_config(AutoConfig.from_pretrained(f"{a.ckpt}/encoder"))
    m = FrontierDecisionEngine(enc, head_layers=2)
    m.load_state_dict(load_file(f"{a.ckpt}/model.safetensors"), strict=True)
    m.eval()
    B, L, K = 1, 64, 3
    args = (torch.ones(B, L, dtype=torch.long) * 10, torch.ones(B, L, dtype=torch.long),
            torch.tensor([[5, 10, 15]]), torch.ones(B, K, dtype=torch.bool), torch.tensor([0]))
    out = f"{a.ckpt}/model.onnx"
    torch.onnx.export(m, args, out,
                      input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
                      output_names=["choice_logits", "act_logits", "score", "noul_logit"],
                      dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
                                    "marker_pos": {0: "B", 1: "K"}, "marker_mask": {0: "B", 1: "K"},
                                    "qtype": {0: "B"}},
                      opset_version=17)
    print("exported", out)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    o = s.run(None, {"input_ids": np.ones((1, 64), np.int64) * 10,
                     "attention_mask": np.ones((1, 64), np.int64),
                     "marker_pos": np.array([[5, 10, 15]]),
                     "marker_mask": np.array([[True, True, True]]),
                     "qtype": np.array([0], np.int64)})
    print("onnxruntime check OK, choice_logits shape:", np.array(o[0]).shape)


if __name__ == "__main__":
    main()
