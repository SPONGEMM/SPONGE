# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04 AS builder

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PIXI_HOME=/opt/pixi
ARG SPONGE_ENV=dev-cpu
ARG SPONGE_SIMD=none
ARG BUILD_JOBS=4

ENV PATH="${PIXI_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        git \
        patchelf \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="${PIXI_HOME}" bash

WORKDIR /workspace

COPY . .

RUN pixi install -e "${SPONGE_ENV}" \
    && if [ "${SPONGE_SIMD}" = "none" ]; then \
           pixi run -e "${SPONGE_ENV}" configure; \
       else \
           pixi run -e "${SPONGE_ENV}" configure "${SPONGE_SIMD}"; \
       fi \
    && pixi run -e "${SPONGE_ENV}" compile "${BUILD_JOBS}" \
    && pixi run -e "${SPONGE_ENV}" package-conda


FROM ubuntu:24.04

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH="/opt/sponge/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        tar \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY --from=builder /workspace/packaging/outputs/*.conda /tmp/

RUN micromamba create -y -p /opt/sponge -c conda-forge /tmp/*.conda \
    && micromamba clean --all --yes \
    && rm -f /tmp/*.conda

WORKDIR /workdir

ENTRYPOINT ["SPONGE"]
CMD ["--help"]
