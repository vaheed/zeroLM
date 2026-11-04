# syntax=docker/dockerfile:1

# ---- build: Rust runtime (ZeroLM Live) --------------------------------------
FROM rust:1.75-slim-bookworm AS build
WORKDIR /src
COPY rust/ rust/
RUN cargo build --release --locked --manifest-path rust/Cargo.toml

# ---- runtime ---------------------------------------------------------------
FROM debian:bookworm-slim

# Unprivileged user; /data holds life files (mount a volume there to keep them).
RUN useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin zerolm \
 && mkdir -p /data /opt/zerolm && chown zerolm /data

COPY --from=build /src/rust/target/release/zerolm /usr/local/bin/zerolm
COPY weights.zlm /opt/zerolm/weights.zlm

ENV ZEROLM_MODEL=/opt/zerolm/weights.zlm
WORKDIR /data
VOLUME /data
USER zerolm

# Interactive ZeroLM Live session; override the command for one-shot use:
#   docker run --rm zerolm "hello"
ENTRYPOINT ["zerolm"]
CMD ["live", "--tag", "default"]
