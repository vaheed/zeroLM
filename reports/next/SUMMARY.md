# ZeroLM Next - experiment summary

cases `final.jsonl` seed 20260701, 21585 cases, commit `b2cf897a40`, weights sha256 `853115ad260a2b8b`

| system | exact match | semantic acc. | false acceptance | abstention |
|---|---:|---:|---:|---:|
| zerolm_next | 1.0 | 1.0 | 0.0 | 0.08339 |
| symbolic | 1.0 | 1.0 | 0.0 | 0.08339 |
| keyword | 0.23073 | 0.26288 | 0.56444 | 0.04091 |
| lookup | 0.00116 | 0.00116 | 0.0 | 0.99893 |
| legacy_v0.2_c | 0.0 | 0.0 | 0.00667 | 0.99944 |
| random weights (max of 5 seeds) | 0 | | | |
| shuffled weights (max of 5 seeds) | 0 | | | |

| ablation | exact match | drop |
|---|---:|---:|
| recurrence | 0.0 | 1.0 |
| state_read | 0.03776 | 0.96224 |
| binding | 0.0 | 1.0 |
| pointer_shift | 0.26652 | 0.73348 |
| time_order | 0.35679 | 0.64321 |
| zero:A_w | 0.26652 | 0.73348 |
| zero:A_o | 0.26652 | 0.73348 |
| zero:B_w | 0.0 | 1.0 |
| zero:B_o | 0.0 | 1.0 |
| zero:S | 0.21648 | 0.78352 |
| zero:E | 0.0 | 1.0 |
| zero:V_w | 1.0 | 0.0 |
| zero:V_o | 1.0 | 0.0 |
| zero:B_e | 1.0 | 0.0 |
| zero:ex_w | 0.0 | 1.0 |
| zero:tf_w | 1.0 | 0.0 |
| zero:rank_eps | 1.0 | 0.0 |
| zero:Gm_w | 0.94359 | 0.05641 |
| zero:Pe_w | 0.0 | 1.0 |
| permute:E:1 | 0.0 | 1.0 |
| permute:B_o:1 | 0.0 | 1.0 |
| permute:A_w:1 | 0.5299 | 0.4701 |
| permute:B_w:1 | 0.0 | 1.0 |
| permute:V_w:1 | 0.0 | 1.0 |
| permute:dec.1.w:1 | 0.10639 | 0.89361 |
| permute:S:1 | 0.94102 | 0.05898 |
| shuffle:1 | 0.0 | 1.0 |
| random:1 | 0.0 | 1.0 |

| ops | exact match | mean parse us | p99 us |
|---:|---:|---:|---:|
| 1 | 1.0 | 233.0 | 474.5 |
| 2 | 1.0 | 748.9 | 1687.8 |
| 4 | 1.0 | 942.7 | 1756.0 |
| 8 | 1.0 | 1693.0 | 3462.5 |
| 12 | 1.0 | 2515.1 | 6877.8 |
| 16 | 1.0 | 3647.7 | 10883.5 |
| 20 | 1.0 | 4704.7 | 9080.2 |
| 22 | 1.0 | 5003.8 | 10191.2 |
| 23 | 1.0 | 4266.7 | 7956.1 |
| 24 | 0.47667 | 4353.7 | 8661.8 |
| 25 | 0.0 | 5139.9 | 15047.8 |
| 28 | 0.0 | 5609.1 | 12813.1 |

**Acceptance floors:** all met
