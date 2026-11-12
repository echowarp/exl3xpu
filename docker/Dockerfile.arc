# exl3xpu Arc B70 fork image: upstream exl3xpu image + newer GPU runtime + this fork's kernel libraries.
# Not built from this file as written (the production image was assembled in layers); the GPU runtime block below is
# the one AllRunner's docker/Dockerfile.arc builds with (packages identical to production's).
#   docker build -f docker/Dockerfile.arc -t exl3xpu-arc .
# The vLLM-side patches of the companion vLLM XPU branch are not part of this image.
FROM ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8
# GPU runtime the kernels were tuned on and production runs: compute-runtime (NEO) 26.31.39395.14 / IGC 2.40.13 /
# Level Zero 1.32 from the kobuk-team/intel-graphics PPA. The PPA has moved on (it now serves 26.40), so the exact
# packages are fetched from Launchpad's archive of superseded builds and checked by sha256. Launchpad often answers 503;
# the download retries. These are the files installed in the production image (dpkg md5sums identical).
ARG NEO_PPA=https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics/+files
# hadolint ignore=DL3008
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    mkdir /tmp/neo; cd /tmp/neo; \
    printf '%s  %s\n' \
      6e79028431b03f98a77d0f0ba56750fcef48cdabbe2b0cd8361d9f03cafb9433 intel-opencl-icd_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      c57e4ca8c780dc85a309e35b4d3249fc866251fea7bf3ebe68a5ab5ef5f71362 libigc2_2.40.13+ds1-1~24.04_amd64.deb \
      e71f41b4dd54c9cda7024680528034204cd911a1c2f644ec2bc3c06e968f5afe libigdfcl2_2.40.13+ds1-1~24.04_amd64.deb \
      2c68fcb150cba1b679063266cdf0054f26f18af90de71d6d3b614eaf95c98f80 libigdgmm12_22.10.1-1~24.04~ppa1_amd64.deb \
      d4001dc474d3cb74ded50a449b9f646cfdfbb31e69e70b8af38f2d18273dfce1 libze1_1.32.0-1~24.04~ppa1_amd64.deb \
      ff2de371d8f1489adf6b71980c929cbbc5ebf08f5a4e64a7572d12ddf264b3d9 libze-dev_1.32.0-1~24.04~ppa1_amd64.deb \
      39f10740e40cf0de1015997d0de094a3c3dc6371c3b9ea1dcd15771d64559c14 libze-intel-gpu1_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      4808c4d96b3938b268371c4fc1336dcad7e3c56b71bd82913f98430067b1e346 libze-intel-gpu-dev_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      > SHA256SUMS; \
    while read -r _ f; do \
      for i in 1 2 3 4 5 6 7 8; do \
        curl -fsSL --max-time 300 -o "$f" "${NEO_PPA}/$(echo "$f" | sed 's/+/%2B/g')" && break; \
        rm -f "$f"; sleep $((i * 10)); \
      done; \
    done < SHA256SUMS; \
    sha256sum -c SHA256SUMS; \
    apt-get install -y --no-install-recommends --allow-downgrades ./*.deb; \
    dpkg -l | grep -E "libze|intel-opencl|libigc"; \
    dpkg-query -W -f '${Version}' libze-intel-gpu1 | grep -qx '26.31.39395.14-1~24.04~ppa1'; \
    cd /; rm -rf /tmp/neo /var/lib/apt/lists/*

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
