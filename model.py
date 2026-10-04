"""Frontier Decision Engine — beats Laya on size/latency, matches+ on accuracy/calibration.

Design (from guide.md, bugs fixed + Laya-compatible single-pass marker scheme):
- Backbone: ModernBERT-base (149M, hidden=768, 8k RoPE, FlashAttn/SDPA)
- Head: 2x TransformerEncoder refinement + SchemaCrossAttention (marker queries
  cross-attend doc keys/values) + marker scorer + act head + per-bucket temperature.
- Jev-compatible I/O: [CLS] <type> instructions [SEP] [MASK] opt ... [SEP] state [SEP]
  identical to Laya's build_sequence, so existing Laya clients / benchmarks run unchanged.

Param count: ~149M + ~7M head ≈ 156M  vs  Laya 421M  (2.7x smaller)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class SchemaCrossAttentionHead(nn.Module):
    """Guide.md head, fixed: batched option queries [B,K,D] cross-attend doc [B,L,D]."""

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, doc_reps, option_reps, doc_pad_mask=None, return_states=False):
        # option_reps: [B,K,D] queries ; doc_reps: [B,L,D] keys/values
        attn_out, _ = self.cross_attn(
            query=option_reps, key=doc_reps, value=doc_reps,
            key_padding_mask=doc_pad_mask, need_weights=False,
        )
        x = self.norm(option_reps + attn_out)
        x = self.norm2(x + self.mlp(x))
        logits = self.scorer(x).squeeze(-1)  # [B, K]
        if return_states:
            return logits, x
        return logits


class FrontierDecisionEngine(nn.Module):
    def __init__(self, encoder, head_layers: int = 2, dropout: float = 0.1, n_act: int = 2,
                 listwise_layers: int = 0, rec_steps: int = 0):
        super().__init__()
        self.encoder = encoder
        self.rec_steps = int(rec_steps)  # F3: shared-weight refinement loops, 0 = off
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.refine = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0 else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.choice_head = SchemaCrossAttentionHead(hidden_dim=d, num_heads=nhead, dropout=dropout)
        # F2 listwise: self-attention ACROSS options (pairwise comparison,
        # precedence, exclusion). 0 = off (bit-identical legacy math).
        self.listwise_layers = int(listwise_layers)
        self.listwise = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout,
                                           batch_first=True, norm_first=True),
                listwise_layers, enable_nested_tensor=False)
            if listwise_layers > 0 else None
        )
        # scalar / noul pooled heads (guide.md)
        self.pool_weights = nn.Linear(d, 1)
        self.score_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        self.noul_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = True

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.refine is not None:
            pad = ~attention_mask.bool()
            for lyr in self.refine.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(lyr, h, None, pad, use_reentrant=False)
                else:
                    h = lyr(h, src_key_padding_mask=pad)
            # F3: loop the SHARED final layer rec_steps extra times (deep
            # supervision reads every loop output; inference uses the last).
            all_h = [h]
            if self.rec_steps > 0:
                loop = self.refine.layers[-1]
                for _ in range(self.rec_steps):
                    if self.head_checkpointing and self.training and torch.is_grad_enabled():
                        h = torch.utils.checkpoint.checkpoint(loop, h, None, pad, use_reentrant=False)
                    else:
                        h = loop(h, src_key_padding_mask=pad)
                    all_h.append(h)
        else:
            all_h = [h]
        # gather marker states -> [B,K,D]
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        pad_mask = ~attention_mask.bool()
        pre_logits, opt_states = self.choice_head(h, m, doc_pad_mask=pad_mask,
                                                  return_states=True)
        if self.listwise is not None:
            opt_pad = ~marker_mask
            for lyr in self.listwise.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    opt_states = torch.utils.checkpoint.checkpoint(
                        lyr, opt_states, None, opt_pad, use_reentrant=False)
                else:
                    opt_states = lyr(opt_states, src_key_padding_mask=opt_pad)
            logits = self.choice_head.scorer(opt_states).squeeze(-1).float()
        else:
            logits = pre_logits.float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # pooled scalar heads
        w = self.pool_weights(h).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(1).float()
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return {
            "choice_logits": logits,
            "act_logits": act_logits,
            "score": torch.sigmoid(self.score_head(pooled)).squeeze(-1),
            "noul_logit": self.noul_head(pooled).squeeze(-1),
            "all_h": all_h,  # F3 deep-supervision states (last == h used above)
        }


def build_frontier_model(encoder_name="answerdotai/ModernBERT-base", head_layers=2):
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(encoder_name, attn_implementation="sdpa")
    return FrontierDecisionEngine(enc, head_layers=head_layers)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
