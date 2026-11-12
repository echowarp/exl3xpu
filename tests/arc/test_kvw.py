"""I4-KVW: fused int4 KV write vs vLLM's reshape_and_cache_int4 (+ the plugin's K smoothing),
byte by byte, plus timing at decode / prefill shapes.  python test_kvw.py exl3xpu/_C.so"""
import sys, time, torch
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4, _get_rht_signs
dev = "xpu"; H, D, BS, NB = 4, 256, 3200, 8
signs = _get_rht_signs(256, 0, torch.device("cpu")).float().contiguous().to(dev)
def views(kv):
    kc, vc = kv.transpose(1, 2).split(132, dim=-1) if False else (None, None)
    return kc, vc
def make():
    kv = torch.zeros(NB, BS, H, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)   # as vLLM: logical (NB,H,BS,264)
    kc, vc = kv.transpose(1, 2).split(132, dim=-1)
    raw = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = kv.stride()
    ks = torch.as_strided(raw, (NB, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 128 // 4)
    vs = torch.as_strided(raw, (NB, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 260 // 4)
    return kv, kc, vc, ks, vs
ok = True
for T, smooth in ((8, False), (8, True), (1, True), (3000, True)):
    torch.manual_seed(T)
    k = (torch.randn(T, H, D) * 2).half().to(dev); k[:, :, 7] += 6.0
    v = torch.randn(T, H, D).half().to(dev)
    m = (torch.randn(H, D) * 0.5).float().to(dev)
    slots = torch.randperm(NB * BS, device=dev)[:T].long(); slots[::5] = -1 if T > 4 else slots[::5]
    kv1, kc1, vc1, ks1, vs1 = make(); kv2, kc2, vc2, _, _ = make()
    kin = (k - m.half()) if smooth else k
    reshape_and_cache_int4(kin, v, kc1, vc1, slots, k_scale_cache=ks1, v_scale_cache=vs1)
    E.exl3_i4_kv_write(k, v, kc2, slots, signs, m if smooth else None)
    torch.xpu.synchronize()
    a, b = kv1.contiguous().view(-1, 264), kv2.contiguous().view(-1, 264)
    rows = (a != b).any(1)
    nz = (a != 0).any(1) | (b != 0).any(1)
    nbad = int(rows.sum()); ntot = int(nz.sum())
    print(f"T={T} smooth={smooth}: rows written {ntot}, rows differing {nbad}", flush=True)
    if nbad:
        i = int(rows.nonzero()[0]); dd = (a[i] != b[i]).nonzero().flatten()[:8].tolist()
        print("   first differing row bytes", dd, a[i][dd].tolist(), b[i][dd].tolist())
    ok &= nbad == 0
def timeit(f, n=50):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e3
for T in (8, 64, 8192):
    k = torch.randn(T, H, D).half().to(dev); v = torch.randn_like(k); m = torch.randn(H, D).float().to(dev)
    slots = torch.arange(T, device=dev).long()
    kv1, kc1, vc1, ks1, vs1 = make()
    t_ref = timeit(lambda: reshape_and_cache_int4(k - m.half(), v, kc1, vc1, slots, k_scale_cache=ks1, v_scale_cache=vs1))
    t_new = timeit(lambda: E.exl3_i4_kv_write(k, v, kc1, slots, signs, m))
    print(f"T={T}: vLLM write + smoothing {t_ref:.3f} ms, fused {t_new:.3f} ms", flush=True)
print("KVW", "BIT-EXACT" if ok else "MISMATCH")
