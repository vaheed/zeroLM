# Installing and running ZeroLM

There are three ways to run ZeroLM. Pick the one that fits.

| You want | Use | Needs |
| --- | --- | --- |
| The original stateless chat, tests, and model editing | [C runtime](#1-c-runtime) | C11 compiler, `make`, Python 3.10+ |
| ZeroLM Live (memory, teaching, JSON-lines mode) | [Rust port](#2-rust-port) | Rust (tested with 1.75) |
| ZeroLM Live without installing toolchains | [Docker](#3-docker) | Docker with the Compose plugin |
| Other programs calling ZeroLM over HTTP (OpenAI-compatible) | [HTTP API](#4-http-api) | the Rust build, or Docker |

The C and Rust runtimes read the same `weights.zlm` and give identical answers for
one-shot prompts (CI checks this on every change).

## 1. C runtime

```bash
git clone https://github.com/vaheed/zeroLM.git
cd zeroLM
make                      # builds ./zerolm from main.c and weights.zlm
./run.sh                  # interactive chat (builds first if needed)
./zerolm "Who are you?"   # one prompt, no session
make pipeline             # rebuild weights, build, run every test and the benchmark
```

Requirements: `cc` (GCC or Clang), GNU Make, and Python 3.10+ for the scripts (standard
library only). On macOS run `xcode-select --install` first if the compiler is missing.

Windows: `run.bat` works with MinGW-w64 GCC. The full `make pipeline` workflow targets
Linux and macOS.

## 2. Rust port

```bash
cd rust
cargo build --release     # produces rust/target/release/zerolm
cargo test                # unit tests + end-to-end tests of zerolm live
cargo clippy --all-targets -- -D warnings
```

The crate is standard-library only. On Ubuntu, `apt install cargo rustc rust-clippy` is enough.

Run it from the repository root so it finds `weights.zlm`, or set `ZEROLM_MODEL`:

```bash
rust/target/release/zerolm "Who are you?"                    # same answer as the C build
rust/target/release/zerolm live --tag me                     # ZeroLM Live, memory in life.me.zlm
rust/target/release/zerolm live --jsonl --tag me             # JSON in, JSON out, one line per turn
ZEROLM_MODEL=/path/to/weights.zlm rust/target/release/zerolm live
```

Live options: `--tag NAME` or `--life FILE` (not both), `--capacity N` facts (default 64),
`--context BYTES` of transcript (default 16384), `--model FILE`. See
[docs/LIVE.md](docs/LIVE.md) for what you can say to it.

## 3. Docker

```bash
docker compose build
docker compose run --rm zerolm                      # interactive ZeroLM Live (tag "default")
docker compose run --rm zerolm live --tag work      # a separate session with its own memory
docker compose run --rm zerolm "does it use training"   # one-shot, nothing stored
docker compose up -d zerolm-daemon                  # optional long-running JSON-lines service
```

Notes:

- Memory lives in the named volume `zerolm-data`, one `life.<tag>.zlm` per tag. Remove it
  with `docker compose down -v`.
- The container runs as an unprivileged user, with a read-only root filesystem, no
  network (`network_mode: none`), all capabilities dropped, and a 256 MB memory limit.
  The image is about 117 MB.
- Two containers cannot use the same tag at the same time: the second one exits with an
  error that points to `--tag`. This holds across containers because the lock is an OS
  file lock on the shared volume.
- The JSON-lines service reads requests on stdin and answers on stdout; there is no network
  port and no HTTP server. Talk to the daemon with
  `docker attach --sig-proxy=false <container>` (detach with Ctrl-p Ctrl-q), or skip the
  daemon and pipe lines through
  `docker compose run --rm -T zerolm live --jsonl --tag NAME`.
- `scripts/docker_smoke.sh` builds the image and checks all of the above end to end.

## 4. HTTP API

```bash
rust/target/release/zerolm serve                                   # http://127.0.0.1:8080
ZEROLM_TOKEN=a-long-secret-token rust/target/release/zerolm serve --addr 0.0.0.0:8080
ZEROLM_TOKEN=a-long-secret-token docker compose up -d zerolm-api   # host port 8080, or set ZEROLM_PORT
curl localhost:8080/v1/health
```

It refuses to listen on a non-loopback address without a token of at least 16 characters, and has no
TLS (put a reverse proxy in front to expose it). Session files go in `--dir` (the `zerolm-data` volume
in Docker). Full reference: [docs/API.md](docs/API.md).

## Checking an install

```bash
./zerolm --self-test                          # C: 48 checks should pass
rust/target/release/zerolm --self-test        # Rust: same output
python3 scripts/evaluate.py                   # labelled prompts, expects 100%
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `model error: cannot open model file: weights.zlm` | Run from the repository root, pass `--model FILE`, or set `ZEROLM_MODEL`. |
| `life file ... is in use by another session` | Another `zerolm live` has that life file open. Use a different `--tag`, or stop the other session. The lock is an OS file lock, so a crashed or killed session releases it automatically. |
| `life error: invalid life header` | The life file is not a ZeroLM life file or is corrupt. It is left untouched; move it away or restore a copy. |
| `make: cc: command not found` | Install a C compiler (`build-essential` on Debian/Ubuntu, Xcode tools on macOS). |
| Docker: `permission denied` on `/data` | Use the named volume from `docker-compose.yml`; a bind mount must be writable by uid 10001. |
