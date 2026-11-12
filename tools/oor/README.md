# torch XPU allocator cap (OOR guard)

`llmt_memprobe.py` is loaded into every Python process by `llmt_memprobe.pth` (copy both into the environment's
site-packages). It starts a daemon thread that does nothing until `torch.xpu` is initialized in that process, then:

- `TORCH_CAP_MIB=N`: caps the torch XPU caching allocator at N MiB (`torch.xpu.set_per_process_memory_fraction`), so
  it frees cached blocks and retries at the cap instead of taking the last device memory that the Level Zero driver
  needs for its own allocations (which otherwise fail with `UR_RESULT_ERROR_OUT_OF_RESOURCES`);
- every `MEMPROBE_S` seconds (default 10) logs allocator numbers to stderr (`MEMPROBE` lines: allocated, reserved,
  peak, device free, segments, retries, OOMs; numbers only). `MEMPROBE=0` disables the thread.

Production (Arc B70, 32 GB): `TORCH_CAP_MIB=29100 MEMPROBE_S=30`, together with
`PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.85` and vLLM `gpu_memory_utilization=0.94`.
The cap alone, or cap + allocator config without the lower utilization, was not enough: all three parts are needed.
