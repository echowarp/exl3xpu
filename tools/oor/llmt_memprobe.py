"""Every MEMPROBE_S seconds, log torch XPU allocator numbers of the process that owns the
device (EngineCore) to stderr: allocated / reserved / peak reserved / device free per mem_get_info. Loaded by a .pth
line; does nothing until torch.xpu is initialized in this process."""
import os, sys, threading, time
_capped = False


def _cap(t):
    """TORCH_CAP_MIB: cap the torch XPU caching allocator (set_per_process_memory_fraction) so it frees cached blocks and
    retries at the cap instead of taking the last device memory the Level Zero driver needs (UR_RESULT_ERROR_OUT_OF_RESOURCES)."""
    global _capped
    cap = int(os.environ.get("TORCH_CAP_MIB", "0"))
    if _capped or not cap:
        return
    _capped = True
    tot = t.xpu.get_device_properties(0).total_memory
    t.xpu.set_per_process_memory_fraction(min(1.0, (cap << 20) / tot), 0)
    sys.stderr.write(f"MEMPROBE cap: torch allocator capped at {cap} MiB of {tot >> 20} (fraction {(cap << 20) / tot:.4f})\n")


def _run():
    S = float(os.environ.get("MEMPROBE_S", "10"))
    while True:
        time.sleep(S)
        t = sys.modules.get("torch")
        try:
            if t is None or not t.xpu.is_initialized():
                continue
            _cap(t)
            a = t.xpu.memory_allocated() >> 20; r = t.xpu.memory_reserved() >> 20
            pr = t.xpu.max_memory_reserved() >> 20
            try:
                f, tot = t.xpu.mem_get_info(); f >>= 20; tot >>= 20
            except Exception:
                f = tot = -1
            st = t.xpu.memory_stats()
            sys.stderr.write(f"MEMPROBE pid={os.getpid()} t={time.time():.0f} alloc={a} reserved={r} peak_reserved={pr} "
                             f"dev_free={f} dev_total={tot} segs={st.get('segment.all.current', -1)} "
                             f"peak_alloc={st.get('allocated_bytes.all.peak', 0) >> 20} inactive_split={st.get('inactive_split_bytes.all.current', 0) >> 20} "
                             f"alloc_retries={st.get('num_alloc_retries', -1)} ooms={st.get('num_ooms', -1)}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"MEMPROBE error {e!r}\n"); time.sleep(60)
if os.environ.get("MEMPROBE", "1") == "1":
    threading.Thread(target=_run, daemon=True, name="memprobe").start()
