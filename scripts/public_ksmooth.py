#!/usr/bin/env python3
"""Build a PUBLIC K-smoothing table (EXL3_KSMOOTH) for the int4/int8 KV path from public text only.

The table is the robust per-layer / KV-head / channel mean key ({layer name: float32 [H, D]}), measured by the exl3xpu
plugin itself: an EAGER engine started with EXL3_KSMOOTH_CALIB=<file> (and no EXL3_KSMOOTH) accumulates the keys of
every real token it writes to the KV cache and saves the mean after each call (EXL3_KSMOOTH_ROBUST=1: token-heads with a
key norm above 3x the median are left out). This script only feeds that engine public text, prefill only (max_tokens 1),
fresh cache_salt per prompt, and records every input with its sha256 (<out>.inputs.json).

Inputs = the groups and shares of exl3xpu scripts/public_draft_vocab.py (GSM8K / IFBench left out: held-out eval sets):
  code 0.30 Python stdlib *.py (host interpreter), docs 0.25 dist-info METADATA (host packages), tool 0.25 json-mode-eval
  (prompts + schemas + answers, plus tool-call chats rendered by the server's chat template), chat 0.10 MT-Bench +
  HumanEval, prose 0.10 Gutenberg pg1342 / pg2600 / pg2701.

  # 1. start an eager engine (enforce_eager) with EXL3_KSMOOTH_CALIB=/out/public_r.pt, EXL3_KSMOOTH_ROBUST=1 and no
  #    EXL3_KSMOOTH, on the KV dtype you serve with (int4_per_token_head or int8_per_token_head)
  # 2. feed it (--data: the directory with HumanEval.jsonl, mt_bench_question.jsonl, json_mode_eval.jsonl, pg1342.txt,
  #    pg2600.txt, pg2701.txt, the same files as scripts/public_draft_vocab.py)
  python3 -I scripts/public_ksmooth.py --port 8000 --data <public data dir> --tokens 400000 --out /out/public_r.pt
  # 3. stop the engine; serve with EXL3_KSMOOTH=/out/public_r.pt (without EXL3_KSMOOTH_CALIB)
  (--out is only recorded in the inputs file; the engine writes the table to its EXL3_KSMOOTH_CALIB path)

Measured (Arc B70, Qwen3.8-27B EXL3 4.0 bpw, int4 KV, 126,099 held-out GSM8K tokens teacher-forced, vs a table built
the same way on production agent traffic): NLL +0.037 %, top-5 KL 4.2e-3, top-1 agreement 98.27 %; without any table
NLL +0.43 %. Speed: the table enables the fused int4 KV write, about +6-7 % on a fast host and up to +35 % per-stream
decode on a slow, host-bound CPU."""
import argparse, glob, hashlib, json, os, random, sys, sysconfig, uuid, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8000"); ap.add_argument("--data", required=True)
ap.add_argument("--tokens", type=int, default=400000, help="total prompt tokens (approx., 4 chars/token)")
ap.add_argument("--chunk", type=int, default=24000, help="characters per prompt")
ap.add_argument("--out", required=True)
a = ap.parse_args()
B = "http://127.0.0.1:" + a.port
SHARES = {"code": 0.30, "docs": 0.25, "tool": 0.25, "chat": 0.10, "prose": 0.10}
sha = lambda b: hashlib.sha256(b).hexdigest()
groups = {g: [] for g in SHARES}; inputs = []
def post(path, body):
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def add(group, path, texts, note):
    inputs.append({"group": group, "file": os.path.basename(path), "sha256": sha(open(path, "rb").read()), "note": note})
    groups[group].extend(texts)
D = a.data
add("chat", f"{D}/mt_bench_question.jsonl", ["\n\n".join(r["turns"] + r.get("reference", [])) for r in jl(f"{D}/mt_bench_question.jsonl")], "MT-Bench (Apache-2.0)")
add("chat", f"{D}/HumanEval.jsonl", [r["prompt"] + r["canonical_solution"] for r in jl(f"{D}/HumanEval.jsonl")], "HumanEval (MIT)")
for b in ("pg1342", "pg2600", "pg2701"):
    add("prose", f"{D}/{b}.txt", [open(f"{D}/{b}.txt", encoding="utf-8", errors="ignore").read()], "Project Gutenberg, public domain")
model = json.load(urllib.request.urlopen(B + "/v1/models", timeout=30))["data"][0]["id"]
tool = []
for i, r in enumerate(jl(f"{D}/json_mode_eval.jsonl")):
    tool.append("\n".join(m["content"] for m in r["prompt"]) + "\n" + r["schema"] + "\n" + r["completion"])
    try:
        schema, ans = json.loads(r["schema"]), json.loads(r["completion"])
    except Exception:
        continue
    if not isinstance(ans, dict) or i % 4:          # every 4th record also as a rendered tool-call chat
        continue
    name = "get_" + "".join(c if c.isalnum() else "_" for c in str(schema.get("title", f"record_{i}"))).lower()
    params = {k: v for k, v in schema.items() if k in ("type", "properties", "required")}
    tools = [{"type": "function", "function": {"name": name, "description": schema.get("description", "Look up a record in the documentation index."), "parameters": params}}]
    user = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
    msgs = [{"role": "system", "content": "You are a documentation assistant. Use the tools to look things up."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "c%d" % i, "function": {"name": name, "arguments": json.dumps(ans)}}]},
            {"role": "tool", "tool_call_id": "c%d" % i, "content": json.dumps(ans, indent=2)},
            {"role": "assistant", "content": "```json\n" + json.dumps(ans, indent=2) + "\n```"}]
    try:
        tool.append(post("/tokenize", {"model": model, "messages": msgs, "tools": tools, "return_token_strs": False,
                                       "add_generation_prompt": False}).get("tokens"))
    except Exception as e:
        print("template skip", i, e, file=sys.stderr)
add("tool", f"{D}/json_mode_eval.jsonl", tool, "NousResearch/json-mode-eval (Apache-2.0) + server chat-template tool-call renders (token ids)")
for group, pat, note in (("code", os.path.join(sysconfig.get_paths()["stdlib"], "*.py"), "host Python stdlib (PSF)"),
                         ("docs", "/usr/lib/python3/dist-packages/*.dist-info/METADATA", "host packages' PyPI long descriptions")):
    files = sorted(glob.glob(pat)); h = hashlib.sha256()
    for f in files:
        raw = open(f, "rb").read(); h.update(os.path.basename(os.path.dirname(f) if f.endswith("METADATA") else f).encode() + b"\0" + sha(raw).encode() + b"\n")
        groups[group].append(raw.decode("utf-8", errors="ignore"))
    inputs.append({"group": group, "file": pat, "files": len(files), "sha256": h.hexdigest(), "note": note})
# prompts: per group, its share of the token budget; text items are concatenated and cut into chunks, token-id items
# (rendered chats) are packed into prompts of up to chunk/4 tokens
rnd = random.Random(1116); sent = {}
for g, items in groups.items():
    budget = int(a.tokens * SHARES[g]); used = 0
    rnd.shuffle(items)
    texts = [x for x in items if isinstance(x, str)]; idl = [x for x in items if isinstance(x, list)]
    if g == "tool" and idl:                                     # half the tool budget as rendered chats
        cur = []
        for ids in idl:
            if used >= budget // 2: break
            cur += ids
            if len(cur) >= a.chunk // 4:
                r = post("/v1/completions", {"model": model, "prompt": cur, "max_tokens": 1, "temperature": 0, "cache_salt": str(uuid.uuid4())})
                used += r["usage"]["prompt_tokens"]; cur = []
    blob = "\n\n".join(texts)
    pos = 0
    while used < budget and pos < len(blob):
        r = post("/v1/completions", {"model": model, "prompt": blob[pos:pos + a.chunk], "max_tokens": 1, "temperature": 0, "cache_salt": str(uuid.uuid4())})
        used += r["usage"]["prompt_tokens"]; pos += a.chunk
    sent[g] = used; print(g, used, "tokens", flush=True)
json.dump({"out": a.out, "tokens_per_group": sent, "shares": SHARES, "inputs": inputs, "model": model,
           "method": "exl3xpu EXL3_KSMOOTH_CALIB, robust (EXL3_KSMOOTH_ROBUST=1), eager engine, prefill only"},
          open(a.out + ".inputs.json", "w"), indent=1)
print("done", sent)
