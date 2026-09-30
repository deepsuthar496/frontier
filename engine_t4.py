# engine_t4.py — Spec §5: Single T4 GPU execution of System-1 Decision Engine.
# Backbone: ModernBERT-base (149M) + SchemaCrossAttentionHead + score/noul heads.
# Faithful to spec, with 2 minimal runtime fixes noted FIX(x):
#   FIX1: key_padding_mask uses True=ignore (invert attention_mask).
#   FIX2: option_ids supports batched [B,K,Lo] and shared [K,Lo] correctly
#         (spec's h_opt.unsqueeze(0) breaks for B>1).
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SchemaCrossAttentionHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, doc_reps, option_reps, doc_mask=None):
        # doc_reps: [B,L,D], option_reps: [B,K,D], doc_mask: [B,L] True=pad(ignore)
        attn_out, _ = self.cross_attn(
            query=option_reps,
            key=doc_reps,
            value=doc_reps,
            key_padding_mask=doc_mask
        )
        x = self.norm(option_reps + attn_out)
        x = x + self.mlp(x)
        logits = self.scorer(x).squeeze(-1)  # [B,K]
        return logits


class SystemOneDecisionEngine(nn.Module):
    def __init__(self, encoder, hidden_dim: int = 768):
        super().__init__()
        self.encoder = encoder
        self.choice_head = SchemaCrossAttentionHead(hidden_dim=hidden_dim)
        self.pool_weights = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.noul_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, option_ids=None, option_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_doc = outputs.last_hidden_state
        # FIX1: invert mask (True = ignore pad)
        doc_pad = ~attention_mask.bool() if attention_mask is not None else None
        weights = F.softmax(self.pool_weights(h_doc).masked_fill(
            ~attention_mask.bool().unsqueeze(-1) if attention_mask is not None else 0, -1e4
        ), dim=1)
        pooled_doc = torch.sum(h_doc * weights, dim=1)
        score = torch.sigmoid(self.score_head(pooled_doc))
        noul_logit = self.noul_head(pooled_doc) / self.temperature.clamp_min(1e-3)
        results = {"score": score,
                   "noul_prob": torch.sigmoid(noul_logit),
                   "noul_logit": noul_logit}

        if option_ids is not None:
            B = h_doc.size(0)
            if option_ids.dim() == 3:
                # [B,K,Lo] per-example options
                Bo, K, Lo = option_ids.shape
                assert Bo == B, f"option batch {Bo} != doc batch {B}"
                om = option_mask if option_mask is not None else torch.ones_like(option_ids)
                flat_ids = option_ids.reshape(B * K, Lo)
                flat_m = om.reshape(B * K, Lo)
                opt_out = self.encoder(input_ids=flat_ids, attention_mask=flat_m)
                h_opt = opt_out.last_hidden_state[:, 0, :].reshape(B, K, -1)
            elif option_ids.dim() == 2:
                # [K,Lo] shared options -> expand to batch
                opt_out = self.encoder(input_ids=option_ids,
                                       attention_mask=option_mask)
                h_opt = opt_out.last_hidden_state[:, 0, :].unsqueeze(0).expand(B, -1, -1)
            else:
                raise ValueError(f"option_ids must be 2D or 3D, got {option_ids.dim()}D")
            # FIX2: pass [B,K,D] queries (spec's unsqueeze(0) was [1,K,D])
            choice_logits = self.choice_head(h_doc, h_opt, doc_mask=doc_pad)
            results["choice_logits"] = choice_logits / self.temperature.clamp_min(1e-3)
            results["choice_probs"] = F.softmax(results["choice_logits"], dim=-1)

        return results


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ================== EXECUTION ON T4 GPU ==================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Running on: CPU (no CUDA)")

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(
        model_name, attn_implementation="sdpa", torch_dtype=torch.float16
        if torch.cuda.is_available() else None)
    decision_model = SystemOneDecisionEngine(backbone).to(device)
    decision_model.eval()
    print(f"Total params: {count_params(decision_model)/1e6:.1f}M "
          f"(encoder {count_params(backbone)/1e6:.1f}M)")

    sample_contexts = [
        "Urgent: Customer reports database deadlock on production cluster node 3.",
        "Billing question: How can I update the corporate VAT registration number?"
    ]

    inputs = tokenizer(
        sample_contexts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    ).to(device)

    # Warmup (kernels autotune, then stable timing)
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for _ in range(5):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _ = decision_model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs -> P50
    N = 20
    lats = []
    with torch.no_grad():
        for _ in range(N):
            s = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = decision_model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - s) * 1000)
    lats.sort()
    p50 = lats[N // 2]

    # Choice-routing smoke test (shared options [K,Lo])
    opts = ["billing issue", "technical outage", "other"]
    opt_tok = tokenizer(opts, padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out_choice = decision_model(**inputs, option_ids=opt_tok["input_ids"],
                                        option_mask=opt_tok["attention_mask"])

    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0

    print("\n========== T4 Benchmark Results ==========")
    print(f"Batch Size: {len(sample_contexts)}")
    print(f"Output Shape: {list(outputs['score'].shape)}")
    print(f"Latency P50 over {N} runs: {p50:.2f} ms (min {lats[0]:.2f} / max {lats[-1]:.2f})")
    print(f"Per-item: {p50/len(sample_contexts):.2f} ms")
    print(f"Allocated VRAM: {allocated_vram:.2f} MB")
    print(f"score: {outputs['score'].float().cpu().squeeze().tolist()}")
    print(f"noul_prob: {outputs['noul_prob'].float().cpu().squeeze().tolist()}")
    print(f"choice_probs: {out_choice['choice_probs'].float().cpu().tolist()}")
    print("==========================================\n")
