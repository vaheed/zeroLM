# ZeroLM benchmark results

- Date: 2026-11-05
- Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (32 logical CPUs), Linux 6.8.0-147-generic
- Tools: rustc 1.75.0 (82e1608df 2023-12-21) (built from a source tarball); cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- Model: 28 intents, `weights.zlm` 29,873 bytes

All numbers are wall-clock on this machine, in one run. Compare rows with each other, not with numbers from other machines.

## 1. One-shot process (start-up + answer + exit)

200 spawns per binary.

| Runtime | mean ms | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| C, answer a prompt | 12.9728 | 12.7687 | 17.4441 | 19.5439 |
| Rust, answer a prompt | 7.2979 | 7.0222 | 9.1793 | 11.9531 |
| C, load model only (`--info`) | 10.5573 | 9.7212 | 17.0852 | 19.7362 |
| Rust, load model only (`--info`) | 7.6116 | 7.2877 | 10.6235 | 12.2819 |

## 2. Steady-state throughput (one process, prompts piped in, no start-up)

5000 prompts.

| Runtime / mode | turns/s |
| --- | ---: |
| C, interactive | 930.5 |
| Rust, interactive, `--faithful` (same softmax math as C) | 924.4 |
| Rust, interactive (default pointer-walk decoder) | 20,867.2 |

## 3. Resident memory (peak RSS of a running process after one answer)

| Process | peak RSS KiB |
| --- | ---: |
| C, interactive | 2,800 |
| Rust, interactive | 2,664 |

## 4. ZeroLM Live round-trip latency (one request, wait for reply)

`--jsonl` session; each turn also saves the life file atomically with an fsync.

| Scenario | n | mean ms | p50 ms | p95 ms | p99 ms | turns/s | peak RSS KiB | facts at end | life file B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| life file on disk, empty memory | 500 | 3.6893 | 3.323 | 5.5602 | 5.7793 | 271.1 | 2736 | 0 | 17 |
| life file on disk, 64 facts, recalling them | 500 | 3.4231 | 3.3013 | 4.3573 | 5.7686 | 292.1 | 2844 | 64 | 2045 |
| life file on disk, tiny context budget (64 B) | 500 | 3.5325 | 3.403 | 4.4296 | 5.7833 | 283.1 | 2756 | 0 | 17 |
| life file on tmpfs, 64 facts, recalling them | 500 | 0.3095 | 0.2858 | 0.4447 | 0.5913 | 3,231.0 | 2856 | 64 | 2045 |

## Reading these numbers

- One-shot: C 12.9728 ms vs Rust 7.2979 ms mean. Loading the model alone (`--info`) takes C 10.5573 ms and Rust 7.6116 ms, so start-up and model loading, not decoding, are what a one-shot call pays for.
- Steady state: the C runtime and `rust --faithful` run the same softmax decoder (930.5 vs 924.4 turns/s). The default Rust decoder walks successor pointers instead, which is mathematically identical here (output is diffed against C in CI) and gives 20,867.2 turns/s.
- ZeroLM Live: mean 3.4231 ms per turn with the life file on disk. On tmpfs it is 0.3095 ms, so nearly all of the disk figure is the per-turn fsync of the life file, not the model or the memory lookup.
- Memory size barely matters: compare the empty and 64-fact rows. Facts are kept alive in the 64-fact rows by asking about each in rotation (unused facts fade after about 100 turns); the run aborts if fewer than 64 remain at the end.
- Latency rows are lock-step round trips (send one line, wait for its reply), not pipelined throughput.
