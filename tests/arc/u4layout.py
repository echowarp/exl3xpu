"""Find the s8 x u4 DPAS B-operand layout: A = ones in row 0 col k only; B = one nibble set -> which (k, n) it hits"""
import sys, torch
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
dev = "xpu"
res = {}
for bit_word in range(64):
    for nib in range(8):
        b = torch.zeros(64, dtype=torch.int64); b[bit_word] = 1 << (4 * nib)
        bw = b.to(torch.int32).to(dev) if b[bit_word] < 2**31 else (b - 2**32).to(torch.int32).to(dev)
        # A: row r has value (k+1) at column k for all k -> C[r, n] = sum_k A[r,k] B[k,n] = (k+1) of the hit k
        a = torch.zeros(8, 32, dtype=torch.int8)
        for r in range(8): a[r] = torch.arange(1, 33, dtype=torch.int8)
        c = torch.zeros(8, 16, dtype=torch.int32, device=dev)
        E.exl3_dpas_u4_probe(a.to(dev).contiguous(), bw.contiguous(), c)
        nz = (c[0] != 0).nonzero().flatten().tolist()
        res[(bit_word, nib)] = [(int(c[0, n].item()) - 1, n) for n in nz]
# print the map: word w nibble j -> (k, n)
for w in range(0, 64, 1):
    print(w, [res[(w, j)] for j in range(8)])
