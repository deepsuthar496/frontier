"""Jev-compatible inference wrapper (mirrors laya RLAgent.system_one)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import QTYPES, build_sequence, confidence_from_probs, temp_bucket


class FrontierAgent:
    def __init__(self, model_dir, device=None):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        with open(f"{model_dir}/frontier_config.json") as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        from model import FrontierDecisionEngine
        from transformers import AutoConfig, AutoModel
        ecfg = AutoConfig.from_pretrained(f"{model_dir}/encoder")
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        self.model = FrontierDecisionEngine(enc, head_layers=self.cfg.get("head_layers", 2))
        self.model.load_state_dict(load_file(f"{model_dir}/model.safetensors"), strict=True)
        self.model.to(self.device).eval()
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.dtype = torch.float16  # T4-safe (bf16 model evaluated in fp16, like Laya)

    @staticmethod
    def _to_internal(qdef):
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        return {"t": t,
                "ins": qdef["instructions"] if isinstance(qdef["instructions"], str)
                else json.dumps(qdef["instructions"]),
                "crit": crit}

    @torch.no_grad()
    def system_one(self, state, questions):
        from common import render_options
        ids = list(questions.keys())
        items = []
        ev_cfg = self.cfg.get("evidence") or {}
        for qid in ids:
            q = self._to_internal(questions[qid])
            st = state
            if ev_cfg.get("enable", False):
                from evidence import select_evidence
                st, info = select_evidence(
                    self, state, questions[qid],
                    budget_tokens=ev_cfg.get("budget_tokens", self.cfg["max_len"]),
                    chunk_tokens=ev_cfg.get("chunk_tokens", 200),
                    overlap=ev_cfg.get("overlap", 50),
                    top_k=ev_cfg.get("top_k", 4),
                    strategy=ev_cfg.get("strategy", "v2"))
                self._last_evidence = getattr(self, "_last_evidence", {})
                self._last_evidence[qid] = info
            seq, markers = build_sequence(self.tok, st, q,
                                          self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r}: options do not fit in head_max_len")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        from common import collate
        b = collate([[it] for it in items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
            out = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device),
                             b["marker_pos"].to(self.device), b["marker_mask"].to(self.device),
                             b["qtype"].to(self.device))
        logits = out["choice_logits"].float().cpu().numpy()
        act = torch.softmax(out["act_logits"].float(), -1).cpu().numpy()
        answers = {}
        n_tokens = int(b["attention_mask"].sum())
        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            z = logits[r, :k] / self.temperature_by_options.get(
                temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max()); p = p / p.sum()
            ext = {"act_probability": float(act[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": round(confidence_from_probs(p, k), 4),
                                "frontier": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": round(confidence_from_probs(p, k), 4),
                                "frontier": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "frontier": ext}
        return {"model": "frontier", "answers": answers,
                "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
