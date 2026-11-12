"""Cheaper MTP drafter for the Arc (exl3xpu).

DS-SUB (EXL3_DS_SUB=1): draft tokens are sampled from the pruned draft head's compact logits (512 / 128 vocab blocks
x 128 ids) with exl3xpu_C::exl3_draft_sample_sub, instead of a -inf-filled 248,320-wide row + index_copy + the
3-kernel full-vocab sampler. Same distribution. The RNG offset comes from a device counter (graph-safe).

DRAFT-GRAPH (EXL3_DRAFT_GRAPH=1, needs DS-SUB): draft passes 2..k of SpecDecodeBaseProposer.propose (the loop after
the first pass) run as one captured XPU graph per (batch size, passes, greedy/random), replayed with one submit.
With model.yaml's cudagraph_mode FULL_DECODE_ONLY the drafter otherwise runs fully eager (~400 kernel launches
per step at k=6, ~60 us of SYCL runtime each on a Broadwell-class host). The first pass (variable query length after
rejections) stays eager. Conditions not met -> stock loop.
"""
from __future__ import annotations

import inspect
import os
import textwrap

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.exl3_draftgraph")   # vllm.* so INFO reaches the engine log

_DS_SUB = os.environ.get("EXL3_DS_SUB", "0") == "1"
_GRAPH = os.environ.get("EXL3_DRAFT_GRAPH", "0") == "1"
_FAR_FROM = int(os.environ.get("EXL3_DRAFT_FAR_FROM", "2"))
_MAX_GRAPHS = int(os.environ.get("EXL3_DRAFT_GRAPH_MAX", "24"))
_CHECK = os.environ.get("EXL3_DRAFT_GRAPH_CHECK", "0") == "1"   # debug: graph vs stock drafts (greedy, B > 1)

_STATE: dict = {}


def _rng(dev):
    st = _STATE.get("rng")
    if st is None:
        seed = int(torch.randint(0, 2**62, (1,)).item())
        st = _STATE["rng"] = (torch.zeros(1, dtype=torch.int64, device=dev), seed)
    return st


def _head_for_pass(lm, k):
    d = lm.exl3_draft
    far = getattr(lm, "exl3_draft_far", None)
    return far if (far is not None and k >= _FAR_FROM) else d


def _sub_sample(proposer, hidden_states, sampling_metadata):
    """Returns (token_ids int64 [B], probs float32 [B, V] or None for greedy), or None if not applicable."""
    model = proposer.model
    lm = getattr(model, "lm_head", None)
    if lm is None or getattr(lm, "exl3_draft", None) is None:
        return None
    k = getattr(model, "_exl3_pass", 0)
    model._exl3_pass = k + 1
    d = _head_for_pass(lm, k)
    from .vllm_plugin import draft_sub_logits   # DRAFT-Q4 int4 head when built
    sub = draft_sub_logits(lm, d, hidden_states)
    idx = d.get("idx_long")
    if idx is None:
        idx = d["idx_long"] = d["idx"].to(torch.int64).contiguous()
    vocab = model.config.vocab_size
    if sampling_metadata.all_greedy:
        am = sub.argmax(dim=-1)
        return idx.index_select(0, am), None
    B = hidden_states.shape[0]
    probs = torch.zeros(B, vocab, dtype=torch.float32, device=sub.device)
    tok = torch.empty(B, dtype=torch.int64, device=sub.device)
    temp = sampling_metadata.temperature
    temp = (temp if temp.dtype == torch.float32 else temp.float()).contiguous()
    ctr, seed = _rng(sub.device)
    # ids beyond the model vocab (lm_head padding) never get probability: the head blocks only hold real ids
    torch.ops.exl3xpu_C.exl3_draft_sample_sub(sub, idx, temp, probs, tok, ctr, seed, k)
    ctr.add_(1)
    return tok, probs


def _sub_ok(sampling_metadata):
    return sampling_metadata.all_greedy or (sampling_metadata.all_random and not sampling_metadata.generators)


def _patch_sample():
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as P
    orig = P._sample_draft_tokens

    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        if (self.method == "mtp" and _sub_ok(sampling_metadata)
                and (self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy)
                and not self.use_heterogeneous_vocab):
            r = _sub_sample(self, hidden_states, sampling_metadata)
            if r is not None:
                return r
        return orig(self, hidden_states, sampling_metadata)

    P._sample_draft_tokens = _sample_draft_tokens
    logger.info("exl3: DS-SUB compact-vocab draft sampling ON (far head from pass %d)", _FAR_FROM)


# ---------------------------------------------------------------------------------------------------------------
# DRAFT-GRAPH

_LOOP_START = "        for token_index in range(self.num_speculative_tokens - 1):\n"
_LOOP_END = "        # [batch_size, num_speculative_tokens]\n"
_CALL = (
    "        draft_token_ids_list, draft_probs_list = _exl3_draft_loop(\n"
    "            self, locals(), _exl3_stock_loop)\n"
)


def _stock_loop_fn(src_loop: str, g: dict):
    """The original loop body as a function of the propose() locals (eager fallback)."""
    names = ["self", "draft_token_ids_list", "draft_probs_list", "hidden_states", "positions", "common_attn_metadata",
             "batch_size", "input_batch_size", "cudagraph_runtime_mode", "batch_size_across_dp", "block_size",
             "sampling_metadata", "per_layer_attn_metadata"]
    body = "def _exl3_stock_loop(L):\n" + "".join(f"    {n} = L[{n!r}]\n" for n in names)
    body += textwrap.indent(textwrap.dedent(src_loop), "    ")
    body += "    return draft_token_ids_list, draft_probs_list\n"
    ns: dict = {}
    exec(compile(body, "<exl3_stock_loop>", "exec"), g, ns)
    return ns["_exl3_stock_loop"]


class _Cap:
    pass


def _graph_ok(self, L) -> bool:
    sm = L["sampling_metadata"]
    return (self.method == "mtp" and _sub_ok(sm) and L["input_batch_size"] == L["batch_size"]
            and not self.use_heterogeneous_vocab
            and not self.constant_draft_positions and self.eplb_state is None
            and (sm.all_greedy or self._enable_probabilistic_draft_probs)
            and self.num_speculative_tokens >= 2)


def _capture(self, L, key, stock):
    """Capture passes 2..k for this batch size. Runs the loop once under capture with static input buffers."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    B, steps, greedy = key
    cm = L["common_attn_metadata"]
    dev = cm.seq_lens.device
    c = _Cap()
    c.ids = torch.zeros(B, dtype=torch.int64, device=dev)
    hs = L["hidden_states"]
    c.hid = torch.zeros(B, hs.shape[-1], dtype=hs.dtype, device=dev)
    pos = L["positions"]
    c.pos = torch.zeros_like(pos)
    c.seq = torch.zeros_like(cm.seq_lens[:B])
    bt = cm.block_table_tensor[:B]
    c.bt = torch.zeros_like(bt)
    temp = L["sampling_metadata"].temperature
    c.temp = torch.ones(B, dtype=torch.float32, device=dev)
    c.out_ids = torch.zeros(B, steps, dtype=torch.int64, device=dev)
    vocab = self.model.config.vocab_size
    c.out_probs = None if greedy else torch.zeros(B, steps, vocab, dtype=torch.float32, device=dev)
    _fill(c, L)
    import dataclasses
    scm = dataclasses.replace(cm, seq_lens=c.seq, block_table_tensor=c.bt)
    for f in ("_seq_lens_cpu", "_num_computed_tokens_cpu"):
        if hasattr(scm, f):
            setattr(scm, f, None)
    sm = L["sampling_metadata"]
    ssm = sm if greedy else dataclasses.replace(sm, temperature=c.temp)
    block_size = L["block_size"]
    k0 = getattr(self.model, "_exl3_pass", 1)

    def body():
        self.model._exl3_pass = k0
        hidden_states = c.hid
        positions = c.pos
        ids = c.ids
        for t in range(steps):
            input_ids = ids.int()
            positions = self._update_positions_dependent_metadata(positions, scm, B, B, block_size)
            _, plam = self.build_per_group_and_layer_attn_metadata(scm, draft_index=t + 1)
            # same buffers as the stock (non-direct) path: stable addresses inside the graph
            self.input_ids[:B] = input_ids
            self.hidden_states[:B] = hidden_states
            if self.supports_mm_inputs:   # Qwen3.8 is a VL model: the drafter takes embeddings
                self.inputs_embeds[:B] = self.model.embed_input_ids(input_ids)
                kw = {"input_ids": None, "positions": self._get_positions(B), "inputs_embeds": self.inputs_embeds[:B]}
            else:
                kw = {"input_ids": self.input_ids[:B], "positions": self._get_positions(B), "inputs_embeds": None}
            if self.pass_hidden_states_to_model:
                kw["hidden_states"] = self.hidden_states[:B]
            with set_forward_context(plam, self.vllm_config, num_tokens=B, num_tokens_across_dp=None,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                     slot_mapping=self._get_slot_mapping(B)):
                ret = self.model(**kw)
                if not self.model_returns_tuple():
                    last, hidden_states = ret, ret
                else:
                    last, hidden_states = ret
            hidden_states = hidden_states[:B]
            tok, probs = self._sample_draft_tokens(last[:B], ssm)
            c.out_ids[:, t].copy_(tok)
            if probs is not None:
                c.out_probs[:, t].copy_(probs)
            ids = tok

    # warm-up (Inductor/Triton compile, allocator) on the side stream, then capture
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        _fill(c, L)
        body()
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    _fill(c, L)
    g = torch.xpu.XPUGraph()
    pool = _STATE.setdefault("pool", torch.xpu.graph_pool_handle())
    with torch.xpu.graph(g, pool=pool):
        body()
    c.g = g
    c.k_after = self.model._exl3_pass
    torch.xpu.synchronize()
    logger.info("exl3: DRAFT-GRAPH captured B=%d passes=%d %s", B, steps, "greedy" if greedy else "random")
    return c


def _fill(c, L):
    B = c.ids.shape[0]
    c.ids.copy_(L["draft_token_ids_list"][-1][:B])
    c.hid.copy_(L["hidden_states"][:B])
    c.pos.copy_(L["positions"])
    cm = L["common_attn_metadata"]
    c.seq.copy_(cm.seq_lens[:B])
    c.bt.copy_(cm.block_table_tensor[:B])
    sm = L["sampling_metadata"]
    if sm.temperature is not None and not sm.all_greedy:
        c.temp.copy_(sm.temperature[:B])


def _why_not(self, L):
    sm = L["sampling_metadata"]
    for name, ok in (("method", self.method == "mtp"), ("sampling", _sub_ok(sm)),
                     ("padded", L["input_batch_size"] == L["batch_size"]), ("hetvocab", not self.use_heterogeneous_vocab), ("constpos", not self.constant_draft_positions),
                     ("eplb", self.eplb_state is None),
                     ("probs", sm.all_greedy or self._enable_probabilistic_draft_probs),
                     ("k", self.num_speculative_tokens >= 2)):
        if not ok:
            return name
    return "?"


def _count(kind, self=None, L=None):
    st = _STATE.setdefault("stats", {"replay": 0, "stock": 0, "n": 0, "why": {}})
    st[kind] += 1; st["n"] += 1
    if kind == "stock" and L is not None:
        w = _why_not(self, L); st["why"][w] = st["why"].get(w, 0) + 1
    if st["n"] % 2000 == 0:
        logger.info("exl3: DRAFT-GRAPH last 2000 loops: %d replayed, %d eager %s, %d graphs", st["replay"], st["stock"],
                    st["why"], len(_STATE.get("caps", {})))
        st.update(replay=0, stock=0, why={})


def _exl3_draft_loop(self, L, stock):
    if not _GRAPH or not _graph_ok(self, L):
        _count("stock", self, L)
        return stock(L)
    B = L["batch_size"]
    steps = self.num_speculative_tokens - 1
    greedy = bool(L["sampling_metadata"].all_greedy)
    key = (B, steps, greedy)
    caps = _STATE.setdefault("caps", {})
    c = caps.get(key)
    if c is None:
        if key in _STATE.setdefault("failed", set()) or len(caps) >= _MAX_GRAPHS:
            return stock(L)
        k_in = self.model._exl3_pass
        try:
            caps[key] = _capture(self, L, key, stock)
        except Exception as e:  # noqa
            _STATE["failed"].add(key)
            logger.warning("exl3: DRAFT-GRAPH capture failed for %s (%r); eager loop for this shape", key, e)
        # the capture step itself uses the stock loop (its first replay right after capture was stale for one row in
        # 1 of ~650 checks); L's inputs are copies, untouched by the warm-up/capture (they ran on static buffers)
        torch.xpu.synchronize()
        self.model._exl3_pass = k_in
        return stock(L)
    _count("replay")
    cm = L["common_attn_metadata"]
    if _CHECK and B > 1:
        seq0 = cm.seq_lens[:B].clone()
        k_in = self.model._exl3_pass
    _fill(c, L)
    c.g.replay()
    self.model._exl3_pass = c.k_after
    if _CHECK and B > 1 and not greedy:
        # random mode: the first graph pass's probs depend only on its inputs (not the RNG): compare with stock's
        g0 = c.out_probs[:, 0].clone()
        cm.seq_lens[:B].copy_(seq0)
        self.model._exl3_pass = k_in
        _, refp = stock(dict(L, draft_token_ids_list=list(L["draft_token_ids_list"]),
                             draft_probs_list=list(L["draft_probs_list"] or [])))
        self.model._exl3_pass = c.k_after
        n0 = len(L["draft_probs_list"] or [])
        d = (refp[n0][:B] - g0).abs().amax(1).tolist()
        st = _STATE.setdefault("checkr", {"n": 0, "bad": 0})
        st["n"] += 1; st["bad"] += int(max(d) > 1e-3)
        if st["n"] % 50 == 0 or (st["n"] < 10 and max(d) > 1e-3):
            logger.info("exl3: DRAFT-GRAPH-CHECK random B=%d: %d checks, %d with max|dprob| > 1e-3; last per-row %s", B,
                        st["n"], st["bad"], [round(x, 5) for x in d])
    if _CHECK and greedy and B > 1:
        # debug: run the stock loop on the same inputs and compare the greedy drafts column by column
        cm.seq_lens[:B].copy_(seq0)
        self.model._exl3_pass = k_in
        ref, _ = stock(dict(L, draft_token_ids_list=list(L["draft_token_ids_list"]),
                            draft_probs_list=None))
        self.model._exl3_pass = c.k_after
        refm = torch.stack([r[:B] for r in ref[1:]], 1)
        diff = (refm != c.out_ids).sum(0).tolist()
        st = _STATE.setdefault("check", {"n": 0, "bad": [0] * steps})
        st["n"] += 1
        for t, d in enumerate(diff):
            if t < len(st["bad"]): st["bad"][t] += int(d > 0)
        if st["n"] % 50 == 0 or (st["n"] < 10 and any(diff)):
            logger.info("exl3: DRAFT-GRAPH-CHECK B=%d: %d checks, columns differing %s; last ref %s graph %s", B,
                        st["n"], st["bad"], refm.tolist(), c.out_ids.tolist())
    cm.seq_lens[:B].copy_(c.seq)   # keep the caller-visible state as the eager loop leaves it
    ids = L["draft_token_ids_list"] + [c.out_ids[:, t].clone() for t in range(steps)]
    dpl = L["draft_probs_list"]
    if dpl is not None and c.out_probs is not None:
        dpl = dpl + [c.out_probs[:, t] for t in range(steps)]
    return ids, dpl


def _patch_graph():
    import vllm.v1.spec_decode.llm_base_proposer as m
    P = m.SpecDecodeBaseProposer
    src = textwrap.dedent(inspect.getsource(P.propose))
    src = textwrap.indent(src, "    ")
    i, j = src.find(_LOOP_START), src.find(_LOOP_END)
    if i < 0 or j < 0 or j < i:
        logger.warning("exl3: DRAFT-GRAPH: propose() loop markers not found; not installed")
        return
    loop_src = src[i + len(_LOOP_START):j]
    g = dict(m.__dict__)
    stock = _stock_loop_fn(_LOOP_START + loop_src, g)
    new = src[:i] + _CALL + src[j:]
    g["_exl3_draft_loop"] = _exl3_draft_loop
    g["_exl3_stock_loop"] = stock
    ns: dict = {}
    exec(compile("class _X:\n" + new, "<exl3_propose>", "exec"), g, ns)
    P.propose = ns["_X"].propose
    logger.info("exl3: DRAFT-GRAPH installed (captured draft passes 2..k, max %d graphs)", _MAX_GRAPHS)


    if _DS_SUB or _GRAPH:
        try:
            _patch_sample()
        except Exception as e:  # noqa
            logger.warning("exl3: DS-SUB not installed (%r)", e)
            return
    if _GRAPH:
        try:
            _patch_graph()
        except Exception as e:  # noqa
            logger.warning("exl3: DRAFT-GRAPH not installed (%r)", e)
