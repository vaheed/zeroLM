# Hardware and runtime notes

## Intended minimum

ZeroLM is designed for a basic computer with:

- 2 CPU cores
- 2 GB RAM
- a C11 compiler to build from source
- no GPU

The model does not need network access or a model-download step. It is single-threaded. The runtime uses fixed arrays and one model allocation. The current model structure is about 674 KiB before process/runtime overhead.

## Real-time expectations

The decoder uses sparse state transitions and a cached output vocabulary. Each recurrent update is O(number of states), not a dense O(number of states squared) matrix multiplication.

`make benchmark` measures the complete interactive process on the current machine. This includes process startup and terminal I/O. CPU type and compiler can change the result, so always measure on the target computer.

## Runtime versus development tools

Runtime requirements:

- `zerolm` executable
- `weights.zlm`

Development requirements:

- C11 compiler
- `make`
- Python 3.10+ standard library

Python is not required for chatting after the C program is compiled.

## Local constrained run

In the build container, the benchmark was also run with CPU affinity restricted to two CPUs and a 2 GB virtual-memory limit. The 1,000-interaction run measured about 0.887 ms per interaction (about 1,128 interactions per second). A separate `/usr/bin/time` check reported 1,280 KiB maximum resident memory for the C executable's `--info` run.

These are measurements from one container host, not a guarantee for every computer. They do show that the current model is far below a 2 GB memory budget and has very low CPU cost on this host.

## Measurements from the benchmark suite

`python3 scripts/benchmark_all.py` records speed and memory for the C runtime, the Rust port and
ZeroLM Live in [reports/benchmarks.md](../reports/benchmarks.md). On the recorded run (a 32-thread
Xeon E5-2697 v4 host, one measurement each):

- **Resident memory:** about 2.7-2.9 MB peak for a running process (C, Rust, and Live with 64
  stored facts), far below the 2 GB target. The `docker-compose.yml` service is capped at 256 MB.
- **Speed:** one-shot calls are 7-13 ms, almost all process start-up and model loading. Inside a
  running process the C runtime answers about 930 prompts/s and the default Rust decoder about
  20,000/s.
- **ZeroLM Live:** about 3.4 ms per turn with the life file on disk, about 0.3 ms on tmpfs. The
  life file is saved with an fsync every turn, so slow or networked disks will be slower. A life
  file with 64 facts is about 2 KB.

The `zerolm-api` Docker service is capped at 256 MB like the others; the API server adds one thread per open
connection (at most 64 by default) and one in-memory session per open tag (at most 256).

The earlier "Local constrained run" above was measured on a previous 18-intent model; the numbers
in the report are the current ones. Locking relies on `flock`, which needs a local filesystem.
