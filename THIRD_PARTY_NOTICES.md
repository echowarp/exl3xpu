# Third-party notices

This fork of exl3xpu is MIT-licensed (`LICENSE`, Copyright (c) 2026 0xSero; the fork's additions are under the same
license). It contains or derives from the following third-party work.

## exllamav3 (MIT)

The EXL3 format, trellis codebooks and Hadamard scheme that the kernels in `csrc/` decode come from exllamav3,
https://github.com/turboderp-org/exllamav3 (as in upstream exl3xpu). Some comments in `csrc/` name exllamav3
equivalents of a technique (e.g. its fp16-accumulation GEMM option).

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```

## vLLM (Apache-2.0)

`exl3xpu/draftgraph.py` (DRAFT-GRAPH) re-creates, at run time, the draft loop of vLLM's
`SpecDecodeBaseProposer.propose` (vllm/v1/spec_decode/llm_base_proposer.py, vLLM 0.26) from vLLM's own source and
mirrors its per-pass steps in the captured graph body. vLLM is https://github.com/vllm-project/vllm, Copyright the
vLLM authors, licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

## Intel oneAPI / oneDNN, PyTorch

The kernels are built with Intel's oneAPI DPC++/C++ compiler and link against oneDNN and PyTorch (torch XPU); none of
their source code is contained here. No Intel sample code is included.
