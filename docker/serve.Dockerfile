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
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
WORKDIR /src
COPY . .
# Registry and target caches survive between builds, so a rebuild after a small change takes about
# a minute instead of a full compile.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p velox-cli --features velox-cli/cuda \
    && cp target/release/velox /usr/local/bin/velox

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
COPY --from=build /usr/local/bin/velox /usr/local/bin/velox
EXPOSE 8010
# The CUDA image's entrypoint (license banner, driver checks) stays outermost.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "velox"]
CMD ["serve", "--help"]
