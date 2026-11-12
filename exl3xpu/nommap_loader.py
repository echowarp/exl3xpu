"""Weight loading without mmap.

vLLM's default safetensors loader memory-maps each shard (safe_open). On a slow root disk shared with swap that turns
a cold start into page faults: measured ~1 MB/s through faults while the disk did 12 MB/s sequential; it also competes with the pinned RAM tiers and shm pools for page cache.
`--safetensors-load-strategy eager` is no fix: it reads a whole 7.9 GiB shard into one bytes object and deserializes
it (~2x the shard in RAM). This iterator parses the safetensors header itself, reads one tensor at a time with
os.preadv (large sequential reads, no mmap) into a private CPU buffer and yields it to vLLM's weight loaders (which copy
it to the GPU). Peak extra RAM = the largest tensor (embedding / lm_head, ~2.5 GB). The read pages stay in the page cache
as ordinary reclaimable cache (shared by both engines, which read the same files, so restarts are warm while RAM allows;
under pressure the kernel drops them cleanly, and no read stalls on a page fault). EXL3_NOMMAP_DROP=1 additionally
drops each read range (posix_fadvise DONTNEED; every start is then a cold read). EXL3_NOMMAP_LOAD=1 (default) enables the
loader, 0 = vLLM's. Engine-agnostic (stdlib + torch).
"""
import json
import os
import struct

import torch

_DROP = os.environ.get("EXL3_NOMMAP_DROP", "0") == "1"
_DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64, "I8": torch.int8,
       "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
       "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def iter_file(path: str):
    """(name, CPU tensor) for every tensor of one safetensors file, in file order, without mmap."""
    fd = os.open(path, os.O_RDONLY)
    try:
        (n,) = struct.unpack("<Q", os.pread(fd, 8, 0))
        header = json.loads(os.pread(fd, n, 8))
        base = 8 + n
        items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                       key=lambda kv: kv[1]["data_offsets"][0])
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        for name, info in items:
            b0, b1 = info["data_offsets"]
            nbytes = b1 - b0
            dtype = _DT[info["dtype"]]
            if nbytes == 0:
                yield name, torch.empty(info["shape"], dtype=dtype)
                continue
            buf = torch.empty(nbytes, dtype=torch.uint8)
            mv = memoryview(buf.numpy())
            got = 0
            while got < nbytes:                      # preadv may return short reads for large ranges
                r = os.preadv(fd, [mv[got:]], base + b0 + got)
                if r <= 0:
                    raise IOError(f"nommap_loader: short read in {path} at {name} ({got} of {nbytes} B)")
                got += r
            if _DROP:
                os.posix_fadvise(fd, base + b0, nbytes, os.POSIX_FADV_DONTNEED)
            yield name, buf.view(dtype).reshape(info["shape"])
    finally:
        os.close(fd)


def safetensors_weights_iterator_nommap(hf_weights_files, use_tqdm_on_load=False, safetensors_load_strategy=None,
                                        local_expert_ids=None, **kw):
    for f in sorted(hf_weights_files):
        yield from iter_file(f)


def patch():
    if os.environ.get("EXL3_NOMMAP_LOAD", "1") != "1":
        return
    import vllm.model_executor.model_loader.default_loader as dl
    import vllm.model_executor.model_loader.weight_utils as wu
    wu.safetensors_weights_iterator = safetensors_weights_iterator_nommap
    dl.safetensors_weights_iterator = safetensors_weights_iterator_nommap
