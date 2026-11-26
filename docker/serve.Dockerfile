# Production image: `velox serve` built from this checkout (the hardware branch decides the tuned
# kernels: build it from `rtx-5090-32g` for an RTX 5090 32 GB).
#
# Build from GitHub without a checkout (what qwen-llm-inferencer's compose does):
#   docker build -f docker/serve.Dockerfile -t velox:rtx-5090-32g \
#     https://github.com/Oceania2018/velox-inferer.git#rtx-5090-32g
# Run (weights and template mounted; see `velox serve --help` for flags):
#   docker run --gpus all -p 8010:8010 -v ninfer-models:/models:ro velox:rtx-5090-32g \
#     serve --artifact /models/qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer --port 8010 --model-id qwen3.8-27b
ARG CUDA_VERSION=13.1.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential git \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
WORKDIR /src
COPY . .
# Registry and target caches survive between builds, so a rebuild after a small change takes about
# a minute instead of a full compile.
#
# Cargo (and build.rs rerun-if-changed) decides freshness by mtime, and a git-URL build context
# stamps every file with its commit time. Building a ref committed before the cached artifacts were
# made would otherwise reuse them and ship another ref's binary. So the cache keeps the sha256 of
# every source file of the last build, and files whose content differs from it are touched first.
# The list is written before compiling, so a failed build cannot leave artifacts it does not cover.
#
# Build identity (`GET /version`, `velox_build_info`): the commit comes from .git, which a git-URL
# context only keeps with the build arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 (qwen-llm-inferencer's
# compose passes it); VELOX_GIT_SHA overrides it, VELOX_GIT_REF names the branch the build asked for.
ARG VELOX_GIT_SHA=
ARG VELOX_GIT_REF=
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    export VELOX_GIT_SHA="${VELOX_GIT_SHA:-$(git rev-parse HEAD 2>/dev/null)}" \
        VELOX_GIT_REF="${VELOX_GIT_REF:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null | grep -v '^HEAD$')}" \
        VELOX_BUILD_UNIX="$(date +%s)" \
    && echo "building velox ${VELOX_GIT_SHA:-unknown} (${VELOX_GIT_REF:-ref unknown})" \
    && find . \( -path ./target -o -path ./.git \) -prune -o -type f -print0 | xargs -0 sha256sum | LC_ALL=C sort > /tmp/new.sha256 \
    && touch target/.src.sha256 && LC_ALL=C sort target/.src.sha256 > /tmp/old.sha256 \
    && LC_ALL=C comm -13 /tmp/old.sha256 /tmp/new.sha256 | cut -c67- | tr '\n' '\0' | xargs -0 -r touch -- \
    && cp /tmp/new.sha256 target/.src.sha256 \
    && cargo build --release --locked -p velox-cli --features velox-cli/cuda \
    && cp target/release/velox /usr/local/bin/velox

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
COPY --from=build /usr/local/bin/velox /usr/local/bin/velox
EXPOSE 8010
# The CUDA image's entrypoint (license banner, driver checks) stays outermost.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "velox"]
CMD ["serve", "--help"]
