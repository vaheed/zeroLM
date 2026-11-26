## Gate A rerun with the mainline online-gradient rule, mode main, outcome **pass**

| arm | final probe BPB (mean) | Delta R (mean) | held-out BPB | novel-context BPB |
|---|---|---|---|---|
| frozen | 4.494 | 0.000 | 8.000 | 8.000 |
| random_features | 4.494 | 0.000 | 4.453 | 4.657 |
| gradient_mainline | 2.948 | 1.546 | 2.708 | 2.980 |
| gradient_perturbed | 4.377 | 0.117 | NaN | NaN |

Byte frequency 4.521 BPB on the same held-out text; Kneser-Ney order 5 2.099.
- P1 passed: value 1.5457, threshold 0.5000. A2 probe gain is significant, at least r, and above the perturbed control
- P2 passed: value 1.8123, threshold 1.0000. A2's end-to-end held-out BPB beats byte frequency by at least the threshold
- P3 passed: value 1.6776, threshold 0.0000. A2 beats A1 end-to-end on novel-context positions
- P4 passed: value 0.9519, threshold 0.5000. retention of the E1 probe gain after E2
- P5 passed: value 0.0059, threshold 0.2500. all values finite, weights bounded, held-out BPB never regresses by more than the threshold between milestones
- V3 passed: value 0.1172, threshold 0.0000. A2-perturbed gain is not significantly positive
