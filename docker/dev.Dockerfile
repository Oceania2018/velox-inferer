# Development image: CUDA toolkit (nvcc) + Rust toolchain.
# Build:  docker build -f docker/dev.Dockerfile -t velox-dev --build-arg BASE=nvidia/cuda:13.1.0-devel-ubuntu24.04 .
# Run:    docker run --rm --gpus all -v "$PWD":/work -w /work velox-dev cargo test --features cuda
ARG BASE=nvidia/cuda:13.1.0-devel-ubuntu24.04
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    --component clippy,rustfmt && chmod -R a+rwX /opt/cargo /opt/rustup
