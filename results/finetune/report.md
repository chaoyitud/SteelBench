# Fine-Tuning Results Report

**Date:** 2026-05-25  
**Task:** Cross-domain regression on steel datasets (SMAPE %, lower is better, mean over 3 seeds)  
**Datasets:** `outo` (outokumpu: `avg_ts` = tensile strength, `avg_ys` = yield strength), `tata` (`rm` = Rm strength, `rp` = Rp0.2 proof strength)

---

## 1. Methods & Hyperparameters

| Method | Base model | Arch keys | Epochs | Steps/epoch | LR | Support | Query | Weight decay | LR schedule |
|---|---|---|---|---|---|---|---|---|---|
| `limix_ft` | LiMix | — | 3 | 100 | 1e-5 | 256 | 64 | 0.01 | warmup+cosine |
| `mitra_ft` | MITRA | — | 5 | 100 | 1e-5 | 512 | 128 | 0.01 | warmup+cosine |
| `tabpfn3_ft` | TabPFN V3 (pip) | 504 | 30 | 100 | 1e-5 | 512 | 128 | 0.01 | warmup+cosine |
| `tabpfn2_ft` | TabPFN V2.6 (pip) | 119 | 30 | 100 | 1e-5 | 512 | 128 | 0.01 | warmup+cosine |
| `tabpfn2_talent_ft` | TabPFN V2 (TALENT local) | 81 | 30 | 100 | 1e-5 | 512 | 128 | 0.01 | warmup+cosine |

Common settings: `n_estimators_finetune = 2`, early stopping patience = 8 epochs (tabpfn models).  
Pool sizes: `open_tata` = 19,303 samples, `open_outo` = 23,301 samples, 132 features.

> **Note:** `tabpfn2_ft` uses pip V2.6 (119-key architecture), which is incompatible with TALENT's local TabPFN V2 evaluator (81-key architecture). The fine-tuned weights cannot be loaded as a TALENT in-context prior, so A3/B3 results for `tabpfn2_ft` are unavailable.
188 x 14
1
The full tree is huge due to data and results_model. Let me show a cleaner top-level view:

Let me print a cleaner summary:

Here's the project tree:



---

## 2. Experiment Protocol

| Experiment | Description |
|---|---|
| **A1** | Fine-tune on `open_tata` pool → evaluate directly on `outo` test set |
| **B1** | Fine-tune on `open_outo` pool → evaluate directly on `tata` test set |
| **A3** | Load A1 checkpoint as prior → run TALENT in-context eval on `outo` splits (`rs50`–`rs80`) |
| **B3** | Load B1 checkpoint as prior → run TALENT in-context eval on `tata` splits (`rs50`–`rs80`) |

`rsN` = N% of data used for training in the TALENT in-context benchmark.

---

## 3. A1/B1 — Cross-Domain Fine-Tuning Benchmark

Fine-tune on one steel corpus and test on the other. Values are SMAPE (%) averaged over 3 seeds.  
Δ = (ft − ZS) / ZS × 100. ZS baseline: `tabpfn_v2` for `tabpfn2_*_ft`; `tabpfn_v3` for `tabpfn3_ft`; no prior ZS for `limix_ft` / `mitra_ft` (they use the A1/B1 protocol directly).

### 3.1 Pool: `open_tata` → Test: `outo`

| Method | outo_uts (%) | outo_ys (%) |
|---|---|---|
| **tabpfn_v2 (ZS)** | 1.0347 | 1.8929 |
| **tabpfn_v3 (ZS)** | 0.8620 | 1.7983 |
| `limix_ft` | **0.8021** | 1.7297 |
| `mitra_ft` | 1.9309 | 4.9021 |
| `tabpfn3_ft` | **0.7688** | **1.6635** |
| `tabpfn2_ft` | 0.7899 | 1.6757 |
| `tabpfn2_talent_ft` | 1.0194 | 1.8948 |

`tabpfn3_ft` Δ vs V3-ZS: −10.8% (uts), −7.5% (ys)  
`tabpfn2_ft` Δ vs V2-ZS: −23.7% (uts), −11.5% (ys)  
`tabpfn2_talent_ft` Δ vs V2-ZS: −1.5% (uts), +0.1% (ys)

### 3.2 Pool: `open_outo` → Test: `tata`

| Method | tata_uts (%) | tata_ys (%) |
|---|---|---|
| **tabpfn_v2 (ZS)** | 1.3611 | 4.0553 |
| **tabpfn_v3 (ZS)** | 1.3069 | 3.9531 |
| `limix_ft` | **1.2893** | **3.8520** |
| `mitra_ft` | 1.7636 | 4.6419 |
| `tabpfn3_ft` | 1.3044 | 3.9809 |
| `tabpfn2_ft` | 1.3102 | 3.9424 |
| `tabpfn2_talent_ft` | 1.3294 | 4.0364 |

`tabpfn3_ft` Δ vs V3-ZS: −0.2% (uts), +0.7% (ys)  
`tabpfn2_ft` Δ vs V2-ZS: −3.7% (uts), −2.8% (ys)  
`tabpfn2_talent_ft` Δ vs V2-ZS: −2.3% (uts), −0.5% (ys)

---

## 4. A3/B3 — TALENT In-Context Evaluation

The A1/B1 checkpoint is loaded as a prior for in-context learning at varying training-set sizes.  
Each cell shows SMAPE (%). ZS = zero-shot with the **same base model** (no fine-tuning). Δ shown per split below each table.

### 4.1 `limix_ft` vs LiMix zero-shot

#### outo (A3, checkpoint from `open_tata`)

| Split | limix-ZS | limix_ft | Δ |
|---|---|---|---|
| outo_avg_ts_rs50 | 0.8218 | 1.0672 | **+29.9%** |
| outo_avg_ts_rs60 | 0.8009 | 1.0639 | **+32.8%** |
| outo_avg_ts_rs70 | 0.7828 | 1.0280 | **+31.3%** |
| outo_avg_ts_rs80 | 0.8154 | 1.0806 | **+32.5%** |
| outo_avg_ys_rs50 | 1.7321 | 2.8134 | **+62.4%** |
| outo_avg_ys_rs60 | 1.7129 | 2.7544 | **+60.8%** |
| outo_avg_ys_rs70 | 1.6437 | 2.6885 | **+63.6%** |
| outo_avg_ys_rs80 | 1.6791 | 2.7162 | **+61.8%** |

#### tata (B3, checkpoint from `open_outo`)

| Split | limix-ZS | limix_ft | Δ |
|---|---|---|---|
| tata_rm_rs50 | 1.3319 | 1.8190 | **+36.6%** |
| tata_rm_rs60 | 1.2737 | 1.7596 | **+38.1%** |
| tata_rm_rs70 | 1.2413 | 1.7338 | **+39.7%** |
| tata_rm_rs80 | 1.2465 | 1.6959 | **+36.1%** |
| tata_rp_rs50 | 3.9188 | 4.3151 | **+10.1%** |
| tata_rp_rs60 | 3.7883 | 4.2068 | **+11.0%** |
| tata_rp_rs70 | 3.8107 | 4.2284 | **+11.0%** |
| tata_rp_rs80 | 3.8642 | 4.2413 | **+9.8%** |

---

### 4.2 `mitra_ft` vs MITRA zero-shot

#### outo (A3, checkpoint from `open_tata`)

| Split | mitra-ZS | mitra_ft | Δ |
|---|---|---|---|
| outo_avg_ts_rs50 | 1.4710 | 1.9083 | **+29.7%** |
| outo_avg_ts_rs60 | 1.4950 | 1.9776 | **+32.3%** |
| outo_avg_ts_rs70 | 1.4823 | 1.8892 | **+27.5%** |
| outo_avg_ts_rs80 | 1.5878 | 1.9919 | **+25.5%** |
| outo_avg_ys_rs50 | 2.8732 | 3.8679 | **+34.6%** |
| outo_avg_ys_rs60 | 2.8958 | 3.8748 | **+33.8%** |
| outo_avg_ys_rs70 | 2.8388 | 3.9114 | **+37.8%** |
| outo_avg_ys_rs80 | 2.9682 | 4.0142 | **+35.2%** |

#### tata (B3, checkpoint from `open_outo`)

| Split | mitra-ZS | mitra_ft | Δ |
|---|---|---|---|
| tata_rm_rs50 | 1.4965 | 1.7570 | **+17.4%** |
| tata_rm_rs60 | 1.4690 | 1.7417 | **+18.6%** |
| tata_rm_rs70 | 1.4048 | 1.7560 | **+25.0%** |
| tata_rm_rs80 | 1.4427 | 1.6963 | **+17.6%** |
| tata_rp_rs50 | 4.3573 | 4.6681 | **+7.1%** |
| tata_rp_rs60 | 4.3091 | 4.6453 | **+7.8%** |
| tata_rp_rs70 | 4.3327 | 4.8847 | **+12.7%** |
| tata_rp_rs80 | 4.3341 | 4.6958 | **+8.3%** |

---

### 4.3 `tabpfn3_ft` vs TabPFN V3 zero-shot

#### outo (A3, checkpoint from `open_tata`)

| Split | tabpfn_v3-ZS | tabpfn3_ft | Δ |
|---|---|---|---|
| outo_avg_ts_rs50 | 0.8620 | 0.9048 | +5.0% |
| outo_avg_ts_rs60 | 0.8164 | 0.8540 | +4.6% |
| outo_avg_ts_rs70 | 0.8016 | 0.8574 | +7.0% |
| outo_avg_ts_rs80 | 0.8172 | 0.8506 | +4.1% |
| outo_avg_ys_rs50 | 1.7983 | 1.9175 | +6.6% |
| outo_avg_ys_rs60 | 1.6852 | 1.8221 | +8.1% |
| outo_avg_ys_rs70 | 1.6112 | 1.7362 | +7.8% |
| outo_avg_ys_rs80 | 1.5870 | 1.7120 | +7.9% |

#### tata (B3, checkpoint from `open_outo`)

| Split | tabpfn_v3-ZS | tabpfn3_ft | Δ |
|---|---|---|---|
| tata_rm_rs50 | 1.3069 | 1.3570 | +3.8% |
| tata_rm_rs60 | 1.2632 | 1.3149 | +4.1% |
| tata_rm_rs70 | 1.2289 | 1.2835 | +4.4% |
| tata_rm_rs80 | 1.2238 | 1.2754 | +4.2% |
| tata_rp_rs50 | 3.9531 | 4.0171 | +1.6% |
| tata_rp_rs60 | 3.8230 | 3.9093 | +2.3% |
| tata_rp_rs70 | 3.7905 | 3.8937 | +2.7% |
| tata_rp_rs80 | 3.8406 | 3.9334 | +2.4% |

---

### 4.4 `tabpfn2_talent_ft` vs TabPFN V2 zero-shot

#### outo (A3, checkpoint from `open_tata`)

| Split | tabpfn_v2-ZS | tabpfn2_talent_ft | Δ |
|---|---|---|---|
| outo_avg_ts_rs50 | 1.0347 | 1.0105 | **−2.3%** |
| outo_avg_ts_rs60 | 1.0014 | 0.9821 | **−1.9%** |
| outo_avg_ts_rs70 | 0.9949 | 0.9794 | **−1.6%** |
| outo_avg_ts_rs80 | 1.0430 | 1.0178 | **−2.4%** |
| outo_avg_ys_rs50 | 1.8929 | 1.8392 | **−2.8%** |
| outo_avg_ys_rs60 | 1.8222 | 1.7508 | **−3.9%** |
| outo_avg_ys_rs70 | 1.7408 | 1.6887 | **−3.0%** |
| outo_avg_ys_rs80 | 1.7066 | 1.6618 | **−2.6%** |

#### tata (B3, checkpoint from `open_outo`)

| Split | tabpfn_v2-ZS | tabpfn2_talent_ft | Δ |
|---|---|---|---|
| tata_rm_rs50 | 1.3611 | 1.3641 | +0.2% |
| tata_rm_rs60 | 1.3199 | 1.3260 | +0.5% |
| tata_rm_rs70 | 1.2598 | 1.2646 | +0.4% |
| tata_rm_rs80 | 1.2633 | 1.2687 | +0.4% |
| tata_rp_rs50 | 4.0553 | 4.0533 | −0.05% |
| tata_rp_rs60 | 3.9570 | 3.9607 | +0.1% |
| tata_rp_rs70 | 3.9489 | 3.9450 | −0.1% |
| tata_rp_rs80 | 4.0217 | 4.0312 | +0.2% |

---

## 5. Summary

### A1/B1 — Direct Evaluation After Fine-Tuning

- **`tabpfn3_ft`** achieves the lowest SMAPE on `outo` (both uts and ys) — clear improvement over V3 zero-shot on the `tata→outo` direction.
- **`tabpfn2_ft`** is comparable to `tabpfn3_ft` on `outo`, with a larger relative improvement over its (weaker) V2 zero-shot baseline.
- **`limix_ft`** is competitive despite far fewer epochs (3 vs 30), and is the best on `tata_ys`.
- **`mitra_ft`** degrades badly vs zero-shot on both targets — fine-tuning appears to hurt MITRA's general representations.
- **`tabpfn2_talent_ft`** shows negligible change vs V2 zero-shot — cross-domain data provides little signal for V2.

### A3/B3 — TALENT In-Context Evaluation

| Method | Direction | Effect vs ZS |
|---|---|---|
| `limix_ft` | both | **Large degradation** (+10% to +64%) |
| `mitra_ft` | both | **Large degradation** (+7% to +38%) |
| `tabpfn3_ft` | both | **Moderate degradation** (+2% to +8%) |
| `tabpfn2_talent_ft` | outo | **Small improvement** (−2% to −4%) |
| `tabpfn2_talent_ft` | tata | **Neutral** (±0.5%) |

Cross-domain fine-tuning consistently **hurts** in-context learning performance for `limix`, `mitra`, and `tabpfn_v3` — the adapted checkpoint no longer functions as a good general prior.  
`tabpfn2_talent_ft` is the only method that shows a consistent (though small) improvement on the `outo` in-context task, suggesting the V2 TALENT architecture is robust enough to partially absorb the domain adaptation without catastrophic forgetting.
