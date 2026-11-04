# Hardware and runtime notes

## Intended minimum

ZeroLM is designed for a basic computer with:

- 2 CPU cores
- 2 GB RAM
- a C11 compiler to build from source
- no GPU

The model does not need network access or a model-download step. It is single-threaded. The runtime uses fixed arrays and one model allocation. The current model structure is about 514 KiB before process/runtime overhead.

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
