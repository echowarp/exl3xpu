"""DRAFT-DYN + context bias for the 27B.

Per request and per draft head (near = exl3_draft, passes 0..FAR_FROM-1; far = exl3_draft_far):
  * EXL3_DRAFT_DYN=E (far head; EXL3_DRAFT_DYN_NEAR=E' for the near head, default E): up to E extra draft-head rows,
    the request's most frequent prompt tokens that are NOT in that static head. Their int4 rows (same per-row g64
    quantization as DRAFT-Q4) are gathered from a host copy of the whole lm_head when the batch rows change.
  * EXL3_DRAFT_BIAS=b: +b on the drafter logit of every prompt token (static rows and extra rows alike).
Exactness with probabilistic drafting (production): the drafter samples from q = softmax over (static rows U extra rows)
of (logits + bias) / T, and the probs handed to rejection sampling are exactly that q scattered to vocab ids
(extra ids are disjoint from the static ids and from each other, pads get -inf -> probability 0). Any proposal q keeps
the target's output distribution exact; only acceptance moves.

This module holds the device-independent pieces (row selection, host-side batch assembly, the sampled-token / probs
fix-up), so they can be tested on the CPU.
"""
from __future__ import annotations

import numpy as np
import torch

NEG_INF = float("-inf")


def static_pos_map(idx: torch.Tensor, n_ids: int) -> np.ndarray:
    """id -> position in the static head (or -1), for ids 0..n_ids-1."""
    pos = np.full(n_ids, -1, dtype=np.int64)
    ix = idx.detach().cpu().numpy().astype(np.int64)
    pos[ix] = np.arange(ix.size, dtype=np.int64)
    return pos


def rows_for_request(prompt_ids, pos: np.ndarray, E: int, vocab: int):
    """-> (extra ids: up to E prompt tokens outside the static head, most frequent first, ties by lower id;
           static positions of the prompt's distinct tokens that are in the static head)."""
    p = np.asarray(prompt_ids if prompt_ids is not None else [], dtype=np.int64)
    p = p[(p >= 0) & (p < min(vocab, pos.size))]
    if p.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    u, c = np.unique(p, return_counts=True)            # u sorted ascending
    sp = pos[u]
    spos = sp[sp >= 0]
    if E <= 0:
        return np.zeros(0, dtype=np.int64), spos
    out_u, out_c = u[sp < 0], c[sp < 0]
    out_u = out_u[np.lexsort((out_u, -out_c))][:E]   # count desc, id asc (deterministic)
    return out_u.astype(np.int64), spos


def assemble(entries, E: int, Ns: int, beta: float, host=None, gs: int = 64, K: int = 0):
    """Batch arrays (numpy) for B = len(entries) rows; entries[i] = (extra_ids, static_pos).
    X [B, E] int64 (pads 0), XB [B, E] fp16 (beta, pads -inf), SB [B, Ns] fp16 (beta at the static positions; None if
    beta == 0), and with host = (packed [V, K/8] int32, scales [V, G] fp16, zeros [V, G] int8): Wq [B*E, K/8] int32,
    Ws [G, B*E] fp16, Wz [G, B*E] int8 (pad rows: zero weights, masked by XB)."""
    B = len(entries)
    X = np.zeros((B, max(E, 0)), dtype=np.int64)
    XB = np.full((B, max(E, 0)), NEG_INF, dtype=np.float16)
    SB = np.zeros((B, Ns), dtype=np.float16) if beta else None
    for i, (ex, spos) in enumerate(entries):
        n = min(len(ex), E)
        if n:
            X[i, :n] = ex[:n]
            XB[i, :n] = beta
        if SB is not None and len(spos):
            SB[i, spos] = beta
    out = dict(X=X, XB=XB, SB=SB)
    if host is not None and E > 0:
        hq, hs, hz = host
        G = hs.shape[1]
        ids = X.reshape(-1)
        valid = (XB.reshape(-1) != NEG_INF)
        Wq = hq[ids].copy()
        Wq[~valid] = 0
        Ws = hs[ids].T.copy()
        Ws[:, ~valid] = 0
        Wz = hz[ids].T.copy()
        Wz[:, ~valid] = 0
        out.update(Wq=Wq, Ws=Ws, Wz=Wz)
    return out


def ext_index(idx_long: torch.Tensor, vocab: int, E: int) -> torch.Tensor:
    """Static ids followed by E virtual ids vocab..vocab+E-1 (one per extra column of a row)."""
    return torch.cat([idx_long, torch.arange(vocab, vocab + E, dtype=torch.int64, device=idx_long.device)])


def finish_sample(tok: torch.Tensor, probs_ext: torch.Tensor, X: torch.Tensor, vocab: int):
    """Map virtual ids (>= vocab) back to the row's extra ids and fold the extra columns of probs_ext [B, vocab+E] into
    the real ids (scatter_add: extra ids are distinct from static ids; pads carry probability 0).
    Returns (tok [B] int64, probs [B, vocab] view)."""
    virt = tok >= vocab
    j = (tok - vocab).clamp(min=0)
    tok = torch.where(virt, X.gather(1, j[:, None]).squeeze(1), tok)
    probs = probs_ext[:, :vocab]
    probs.scatter_add_(1, X, probs_ext[:, vocab:])
    return tok, probs


def finish_greedy(L: torch.Tensor, idx_long: torch.Tensor, X: torch.Tensor):
    Ns = idx_long.numel()
    am = L.argmax(dim=-1)
    return torch.where(am < Ns, idx_long.index_select(0, am.clamp(max=Ns - 1)),
                       X.gather(1, (am - Ns).clamp(min=0)[:, None]).squeeze(1))


@torch.no_grad()
def quantize_rows_int4(w: torch.Tensor, gs: int):
    """Same per-row asymmetric int4 (group gs along K) as vllm_plugin._build_draft_q4. w [n, K] -> packed [n, K/8]
    int32, scales [n, G] fp16, zeros [n, G] int8 (row-major per row; transpose for the aten op's [G, N] layout)."""
    n, K = w.shape
    G = K // gs
    w = w.float().reshape(n, G, gs)
    mn, mx = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-8)
    z = (-mn / s).round().clamp(0, 15)
    q = (w / s + z).round().clamp(0, 15).reshape(n, K).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().view(torch.int32)
    return packed, s.squeeze(-1).half(), z.squeeze(-1).to(torch.int8)


def dequant_rows_int4(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, gs: int):
    """Inverse of quantize_rows_int4 (CPU reference for tests): -> fp32 [n, K]."""
    b = packed.contiguous().view(torch.uint8)
    lo, hi = (b & 15).float(), (b >> 4).float()
    q = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    n, K = q.shape
    q = q.reshape(n, K // gs, gs)
    return ((q - zeros.float()[..., None]) * scales.float()[..., None]).reshape(n, K)
