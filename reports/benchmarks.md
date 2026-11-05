# ZeroLM benchmark results

- Date: 2026-11-05
- Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (32 logical CPUs), Linux 6.8.0-147-generic
- Tools: rustc 1.75.0 (82e1608df 2023-12-21) (built from a source tarball); cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- Model: 28 intents, `weights.zlm` 37,380 bytes

All numbers are wall-clock on this machine, in one run. Compare rows with each other, not with numbers from other machines.

## 1. One-shot process (start-up + answer + exit)

200 spawns per binary.

| Runtime | mean ms | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| C, answer a prompt | 19.9121 | 18.35 | 36.642 | 47.5582 |
| Rust, answer a prompt | 10.4099 | 10.0524 | 15.6318 | 18.0409 |
| C, load model only (`--info`) | 15.8364 | 15.388 | 19.4351 | 33.3988 |
| Rust, load model only (`--info`) | 10.4952 | 9.8422 | 15.7794 | 20.3403 |

## 2. Steady-state throughput (one process, prompts piped in, no start-up)

5000 prompts.

| Runtime / mode | turns/s |
| --- | ---: |
| C, interactive | 715.8 |
| Rust, interactive, `--faithful` (same softmax math as C) | 778.8 |
| Rust, interactive (default pointer-walk decoder) | 8,190.3 |

## 3. Resident memory (peak RSS of a running process after one answer)

| Process | peak RSS KiB |
| --- | ---: |
| C, interactive | 2,928 |
| Rust, interactive | 2,872 |

## 4. ZeroLM Live round-trip latency (one request, wait for reply)

`--jsonl` session; each turn also saves the life file atomically with an fsync.

| Scenario | n | mean ms | p50 ms | p95 ms | p99 ms | turns/s | peak RSS KiB | facts at end | life file B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| life file on disk, empty memory | 500 | 4.4395 | 4.0674 | 6.4674 | 7.6214 | 225.2 | 3068 | 0 | 17 |
| life file on disk, 64 facts, recalling them | 500 | 3.7409 | 3.595 | 4.7879 | 7.2613 | 267.3 | 3084 | 64 | 2045 |
| life file on disk, tiny context budget (64 B) | 500 | 3.73 | 3.6346 | 4.5886 | 5.7391 | 268.1 | 3008 | 0 | 17 |
| life file on tmpfs, 64 facts, recalling them | 500 | 0.3136 | 0.2837 | 0.57 | 0.67 | 3,188.8 | 3124 | 64 | 2045 |

## Reading these numbers

- One-shot: C 19.9121 ms vs Rust 10.4099 ms mean. Loading the model alone (`--info`) takes C 15.8364 ms and Rust 10.4952 ms, so start-up and model loading, not decoding, are what a one-shot call pays for.
- Steady state: the C runtime and `rust --faithful` run the same softmax decoder (715.8 vs 778.8 turns/s). The default Rust decoder walks successor pointers instead, which is mathematically identical here (output is diffed against C in CI) and gives 8,190.3 turns/s.
- ZeroLM Live: mean 3.7409 ms per turn with the life file on disk. On tmpfs it is 0.3136 ms, so nearly all of the disk figure is the per-turn fsync of the life file, not the model or the memory lookup.
- Memory size barely matters: compare the empty and 64-fact rows. Facts are kept alive in the 64-fact rows by asking about each in rotation (unused facts fade after about 100 turns); the run aborts if fewer than 64 remain at the end.
- Latency rows are lock-step round trips (send one line, wait for its reply), not pipelined throughput.
