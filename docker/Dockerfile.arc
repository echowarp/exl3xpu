# exl3xpu Arc B70 fork image: upstream exl3xpu image + newer GPU runtime + this fork's kernel libraries.
# Not built from this file as written (the production image was assembled in layers); treat it as a recipe.
#   docker build -f docker/Dockerfile.arc -t exl3xpu-arc .
# The vLLM-side patches of the companion vLLM XPU branch are not part of this image.
FROM ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8
# GPU runtime the kernels were tuned on: compute-runtime (NEO) 26.31 / IGC 2.40 / Level Zero 1.32
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics && apt-get update \
 && apt-get install -y --only-upgrade libze-intel-gpu1 libze1 intel-opencl-icd libigc2 libigdfcl2 libigdgmm12 libze-intel-gpu-dev libze-dev \
 && dpkg -l | grep -E "libze|intel-opencl|libigc" && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/exl3xpu
COPY pyproject.toml ./
COPY exl3xpu ./exl3xpu
COPY csrc ./csrc
COPY gdnchunk ./gdnchunk
COPY ng ./ng
COPY scripts ./scripts
COPY models ./models
COPY tools ./tools
RUN bash scripts/build_arc.sh all && pip install --no-deps --no-build-isolation -e . \
 && cp tools/oor/llmt_memprobe.py tools/oor/llmt_memprobe.pth "$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"/
ENV VLLM_XPU_USE_SAMPLER_KERNEL=0
ENV MODELS_DIR=/models
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
