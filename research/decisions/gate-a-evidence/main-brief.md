## Representation feasibility (Gate A), mode main, outcome **fail_pivot**

| arm | final probe BPB (mean) | Delta R (mean) | held-out BPB | novel-context BPB |
|---|---|---|---|---|
| frozen | 4.516 | 0.000 | 8.000 | 8.000 |
| random_features | 4.516 | 0.000 | 4.823 | 5.071 |
| local | 4.506 | 0.010 | 4.810 | 5.053 |
| bptt_control | 3.148 | 1.368 | 2.905 | 3.211 |
| local_perturbed | 4.522 | -0.006 | NaN | NaN |

Headroom ratio rho = 0.007. 
- P1 FAILED: value 0.0100, threshold 0.1000. A2 probe gain is significant, at least r, and above the perturbed control
- P2 FAILED: value 0.0073, threshold 0.1000. fraction of the backpropagation gain captured
- P3 passed: value 0.0187, threshold 0.0000. A2 beats A1 end-to-end on novel-context positions
- P4 FAILED: value NaN, threshold 0.5000. retention of the E1 probe gain after E2
- P5 passed: value 0.0159, threshold 0.1000. no non-finite value; late-window degradation bounded
- V2 passed: value 1.3678, threshold 0.5000. backpropagation control gain is significant and at least h
- V3 passed: value -0.0058, threshold 0.0000. A2-perturbed gain is not significantly positive
