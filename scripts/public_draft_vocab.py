"""
Build pruned MTP draft vocabularies (EXL3_DRAFT_VOCAB / EXL3_DRAFT_VOCAB_FAR) from PUBLIC text only.

Same file format and ranking rule as scripts/draft_vocab.py (128-token lm_head blocks ranked by token
frequency, every special / added token's block forced in, top N kept), but with a fixed, documented,
license-clean input set aimed at tool-calling documentation agents, and every input recorded with its
sha256 in the output json ("inputs"), so the files can be rebuilt and audited. One pass writes several
sizes; the smaller sets are prefixes of the same ranking (the far set is a subset of the near set).

Inputs (groups are normalised to fixed shares of the total count, so one large source cannot dominate):
  code  0.30  Python standard library *.py of the running interpreter (PSF license)
  docs  0.25  long descriptions (dist-info METADATA) of the installed Python packages (their own licenses;
              published on PyPI) -- README-style technical documentation
  tool  0.25  json-mode-eval (NousResearch, Apache-2.0): prompts, schemas, JSON answers; plus each schema
              rendered through the model's own chat template as a tool definition, a qwen3_coder tool call
              (arguments = the JSON answer), the tool response and a short answer
  chat  0.10  MT-Bench questions (Apache-2.0), HumanEval prompts + canonical solutions (MIT)
  prose 0.10  Project Gutenberg public-domain books (pg1342, pg2600, pg2701)
GSM8K and IFBench are deliberately NOT inputs: they are the held-out sets for scripts' coverage checks.
Chat scaffolding the model emits constantly gets the same boost as scripts/draft_vocab.py.

  python3 scripts/public_draft_vocab.py --model DIR --data DIR_WITH_PUBLIC_FILES --blocks 512 128 \\
      --out /out/public-{n}.json
  # optional coverage of held-out model outputs (one text per line in a .txt, or jsonl with --eval-key):
  python3 scripts/public_draft_vocab.py ... --eval responses.jsonl:response --compare other-512.json
"""
import argparse, glob, hashlib, json, os, sys, sysconfig
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint dir with tokenizer.json + chat_template.jinja")
ap.add_argument("--data", required=True, help="dir with HumanEval.jsonl, mt_bench_question.jsonl, "
                "json_mode_eval.jsonl, pg1342.txt, pg2600.txt, pg2701.txt")
ap.add_argument("--blocks", type=int, nargs="+", default=[512, 128])
ap.add_argument("--out", required=True, help="output path, {n} = block count")
ap.add_argument("--eval", action="append", default=[], help="FILE.jsonl:key[.idx...] or FILE.txt held-out texts")
ap.add_argument("--compare", action="append", default=[], help="other draft vocab json(s) to score on --eval")
args = ap.parse_args()

SHARES = {"code": 0.30, "docs": 0.25, "tool": 0.25, "chat": 0.10, "prose": 0.10}
tok = AutoTokenizer.from_pretrained(args.model)
V = len(tok)
nblk = (V + 127) // 128
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
sha = lambda b: hashlib.sha256(b).hexdigest()

groups = {g: [] for g in SHARES}          # group -> list of texts
inputs = []                                # provenance records


def add_file(group, path, texts, note=""):
    raw = open(path, "rb").read()
    inputs.append({"group": group, "file": os.path.basename(path), "sha256": sha(raw), "note": note})
    groups[group].extend(texts)


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


D = args.data
# chat: MT-Bench questions, HumanEval
add_file("chat", f"{D}/mt_bench_question.jsonl",
         ["\n\n".join(r["turns"] + r.get("reference", [])) for r in jl(f"{D}/mt_bench_question.jsonl")],
         "MT-Bench questions (lmsys FastChat, Apache-2.0)")
add_file("chat", f"{D}/HumanEval.jsonl",
         [r["prompt"] + r["canonical_solution"] for r in jl(f"{D}/HumanEval.jsonl")], "openai/human-eval (MIT)")
# prose: Gutenberg
for b in ("pg1342", "pg2600", "pg2701"):
    add_file("prose", f"{D}/{b}.txt", [open(f"{D}/{b}.txt", encoding="utf-8", errors="ignore").read()],
             "Project Gutenberg, public domain")
# tool: json-mode-eval + chat-template tool calls built from it
jm = jl(f"{D}/json_mode_eval.jsonl")
tool_texts = []
for i, r in enumerate(jm):
    tool_texts.append("\n".join(m["content"] for m in r["prompt"]) + "\n" + r["schema"] + "\n" + r["completion"])
    try:
        schema, answer = json.loads(r["schema"]), json.loads(r["completion"])
    except Exception:
        continue
    if not isinstance(answer, dict):
        continue
    name = "get_" + "".join(c if c.isalnum() else "_" for c in str(schema.get("title", f"record_{i}"))).lower()
    params = {k: v for k, v in schema.items() if k in ("type", "properties", "required")}
    tools = [{"type": "function", "function": {"name": name, "description": schema.get("description",
              f"Look up a {schema.get('title', 'record')} in the documentation index."), "parameters": params}}]
    user = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
    msgs = [{"role": "system", "content": "You are a documentation assistant. Use the tools to look things up."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": answer}}]},
            {"role": "tool", "content": json.dumps(answer, indent=2)},
            {"role": "assistant", "content": "```json\n" + json.dumps(answer, indent=2) + "\n```"}]
    try:
        tool_texts.append(tok.apply_chat_template(msgs, tools=tools, tokenize=False))
    except Exception as e:  # template rejects a schema shape: keep the plain text only
        print("template skip", i, e, file=sys.stderr)
add_file("tool", f"{D}/json_mode_eval.jsonl", tool_texts,
         "NousResearch/json-mode-eval (Apache-2.0) + chat_template.jinja tool-call renders")
tmpl = os.path.join(args.model, "chat_template.jinja")
if os.path.exists(tmpl):
    inputs.append({"group": "tool", "file": "chat_template.jinja", "sha256": sha(open(tmpl, "rb").read()),
                   "note": "model chat template (renders above)"})
# code: Python stdlib; docs: dist-info METADATA
stdlib = sysconfig.get_paths()["stdlib"]
purelib = sysconfig.get_paths()["purelib"]
for group, pattern, note in (("code", os.path.join(stdlib, "*.py"), f"Python {sys.version.split()[0]} stdlib (PSF)"),
                             ("docs", os.path.join(purelib, "*.dist-info", "METADATA"), "installed packages' PyPI "
                              "long descriptions")):
    files = sorted(glob.glob(pattern))
    h = hashlib.sha256()
    for f in files:
        raw = open(f, "rb").read()
        h.update(os.path.relpath(f, os.path.dirname(pattern.split("*")[0])).encode() + b"\0" + sha(raw).encode() + b"\n")
        groups[group].append(raw.decode("utf-8", errors="ignore"))
    inputs.append({"group": group, "file": pattern, "files": len(files), "sha256": h.hexdigest(),
                   "note": note + " (sha256 over sorted 'relpath\\0sha256' lines)"})

counts = np.zeros(nblk * 128, dtype=np.float64)
n_tok, per_group = 0, {}
for g, texts in groups.items():
    c = np.zeros_like(counts)
    n = 0
    for t in texts:
        ids = enc(t)
        np.add.at(c, ids, 1)
        n += len(ids)
    per_group[g] = n
    n_tok += n
    if n:
        counts += c * (SHARES[g] / n)
counts *= n_tok                                        # back to a token-count scale
for s in ["<|im_start|>assistant\n", "<|im_end|>", "<think>\n", "\n</think>\n\n", "```python\n", "```\n",
          "```json\n", "<tool_call>\n<function=", "</parameter>\n", "</function>\n</tool_call>"]:
    counts[enc(s)] += n_tok / 1000

block_score = counts.reshape(nblk, 128).sum(1)
forced = set()
for t in set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys()):
    if 0 <= t < V:
        forced.add(t // 128)
order = [int(b) for b in np.argsort(-block_score, kind="stable") if b not in forced]
print(f"tokens per group: {per_group} (total {n_tok}); forced blocks {len(forced)}")
outs = {}
for nb in args.blocks:
    chosen = sorted(forced | set(order[: max(0, nb - len(forced))]))
    cov = float(block_score[chosen].sum() / block_score.sum())
    path = args.out.replace("{n}", str(nb))
    json.dump({"model": os.path.basename(os.path.normpath(args.model)), "block_size": 128, "blocks": chosen,
               "n_blocks": len(chosen), "tokens": len(chosen) * 128, "vocab": V, "corpus_tokens": int(n_tok),
               "corpus_coverage": cov, "source": "public", "group_shares": SHARES, "group_tokens": per_group,
               "inputs": inputs, "builder": "exl3xpu scripts/public_draft_vocab.py"}, open(path, "w"))
    outs[path] = chosen
    print(f"{path}: {len(chosen)} blocks ({len(chosen) * 128} of {V} tokens), weighted input coverage {cov:.4f}")

# held-out coverage: fraction of eval tokens whose 128-block is in each vocab (only ratios are printed)
if args.eval:
    ev = []
    for spec in args.eval:
        path, _, key = spec.partition(":")
        if path.endswith(".txt"):
            texts = [l for l in open(path, encoding="utf-8") if l.strip()]
        else:
            texts = []
            for r in jl(path):
                v = r
                for k in key.split("."):
                    v = v[int(k)] if isinstance(v, list) else v[k]
                texts.append(v)
        ids = np.concatenate([np.asarray(enc(t), dtype=np.int64) for t in texts if t]) if texts else np.zeros(0, int)
        ev.append((os.path.basename(path), ids))
    sets = dict((os.path.basename(p), set(b)) for p, b in outs.items())
    for c in args.compare:
        sets[os.path.basename(c)] = set(json.load(open(c))["blocks"])
    for name, ids in ev:
        blk = ids // 128
        row = {k: round(float(np.isin(blk, list(s)).mean()), 4) for k, s in sets.items()}
        print(f"coverage {name} ({len(ids)} tokens): {json.dumps(row)}")
