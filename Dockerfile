# ZeroLM in one image: the toolchain, the source, prebuilt release binaries,
# and prebuilt tests. No model or data is downloaded; the model starts from
# its pinned seeded initialization. See docs/GETTING_STARTED.md.
#
#   docker compose up --build          guided demo with internals shown
#   docker compose run --rm reason     talk to the reasoner yourself
#   docker compose run --rm gates      run every gate and print a summary

# Rust 1.75 is the declared minimum (rust-version) and the CI clippy toolchain.
FROM rust:1.75-slim-bookworm

# clippy is not part of the slim image; the gates need it.
RUN rustup component add clippy

WORKDIR /zerolm
COPY . .

# Release binaries (CLI and phase benchmarks) plus compiled tests, so tests and
# gates start immediately. --locked keeps dependency versions identical to Cargo.lock.
RUN cargo build --release --workspace --bins --locked \
 && cargo test --workspace --locked --no-run \
 && mkdir -p /data reports

# The commit this image was built from (no .git is copied). The Phase 6 gate
# derives its hidden seed from it, exactly as in a git checkout. Declared after
# the build step so a new commit does not invalidate the compile cache.
ARG ZEROLM_COMMIT=unknown
LABEL org.opencontainers.image.revision=$ZEROLM_COMMIT
ENV PATH="/zerolm/target/release:${PATH}" \
    ZEROLM_COMMIT=$ZEROLM_COMMIT \
    ZEROLM_VERBOSE=1

CMD ["scripts/docker/demo.sh"]
