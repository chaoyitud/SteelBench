#!/usr/bin/env python3
"""
Step 6 — Final comparison table generator
Run after all 3-seed final runs and A3/B3 ICL checks are complete.

Usage:
    python finetune/make_comparison_table.py
"""
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Zero-shot baselines (measured: base model, in-domain context, rs50 test set) ──
# Measured with test/train_model_deep.py --seed_num 1 on {outo_avg_ts/ys_rs50, tata_rm/rp_rs50}
# These are the true ZS in-domain baselines on the SAME test sets used in the FT benchmark.
ZS = {
    # site, target, base_model → SMAPE
    ('outo', 'uts', 'limix'):      0.8220,
    ('outo', 'ys',  'limix'):      1.7306,
    ('outo', 'uts', 'mitra'):      1.4710,  # mitra ZS not re-measured (mitra FT skipped)
    ('outo', 'ys',  'mitra'):      2.8732,
    ('outo', 'uts', 'tabpfn_v3'): 0.8557,
    ('outo', 'ys',  'tabpfn_v3'): 1.7957,
    ('outo', 'uts', 'tabpfn_v2'): 1.0304,
    ('outo', 'ys',  'tabpfn_v2'): 1.8814,
    ('tata', 'uts', 'limix'):      1.3307,
    ('tata', 'ys',  'limix'):      3.9148,
    ('tata', 'uts', 'mitra'):      1.7636,
    ('tata', 'ys',  'mitra'):      4.6419,
    ('tata', 'uts', 'tabpfn_v3'): 1.3107,
    ('tata', 'ys',  'tabpfn_v3'): 3.9592,
    ('tata', 'uts', 'tabpfn_v2'): 1.3606,
    ('tata', 'ys',  'tabpfn_v2'): 4.0566,
}

# A3/B3 ICL catastrophic-forgetting check results — hardcoded from final log files
# (fallback when log file cannot be read)
# Format: (pool_tag, model_key, holdout_dataset) → SMAPE
A3_KNOWN = {
    # A3: open_tata ckpt → outo holdout rs70 (uts=tensile, ys=yield strength)
    ('open_tata', 'limix_ft',          'outo_avg_ts_rs70'): 7.81358406e-01,
    ('open_tata', 'limix_ft',          'outo_avg_ys_rs70'): 1.63831953e+00,
    ('open_tata', 'tabpfn3_ft',        'outo_avg_ts_rs70'): 8.04654710e-01,
    ('open_tata', 'tabpfn3_ft',        'outo_avg_ys_rs70'): 1.61888761e+00,
    ('open_tata', 'tabpfn2_talent_ft', 'outo_avg_ts_rs70'): 9.83886408e-01,
    ('open_tata', 'tabpfn2_talent_ft', 'outo_avg_ys_rs70'): 1.68110716e+00,
    # B3: open_outo ckpt → tata holdout rs70 (uts=rm, ys=rp)
    ('open_outo', 'limix_ft',          'tata_rm_rs70'): 1.24044875e+00,
    ('open_outo', 'limix_ft',          'tata_rp_rs70'): 3.80978733e+00,
    ('open_outo', 'tabpfn3_ft',        'tata_rm_rs70'): 1.23718339e+00,
    ('open_outo', 'tabpfn3_ft',        'tata_rp_rs70'): 3.80496851e+00,
    ('open_outo', 'tabpfn2_talent_ft', 'tata_rm_rs70'): 1.25896133e+00,
    ('open_outo', 'tabpfn2_talent_ft', 'tata_rp_rs70'): 3.94301121e+00,
}

MODEL_KEY_TO_BASE = {
    'limix_ft':          'limix',
    'tabpfn3_ft':        'tabpfn_v3',
    'tabpfn2_talent_ft': 'tabpfn_v2',
    'mitra_ft':          'mitra',
}


def parse_smape_from_log(log_path: Path) -> float | None:
    """Parse SMAPE MEAN from test/train_model_deep.py log output.
    Handles both plain decimals and scientific notation (e.g. 7.81358406e-01).
    """
    try:
        text = log_path.read_text()
        vals = re.findall(r'SMAPE\s+MEAN\s*[=:]\s*([\d.eE+\-]+)', text, re.I)
        return float(vals[0]) if vals else None
    except FileNotFoundError:
        return None


def load_final_run_results(results_dir: Path) -> pd.DataFrame:
    """Load all *_results.csv from the final run output directory."""
    dfs = []
    for csv_path in results_dir.glob("*_results.csv"):
        df = pd.read_csv(csv_path)
        dfs.append(df)
    if not dfs:
        print(f"[WARN] No result CSVs found in {results_dir}")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_table(results_dir: Path, logdir: Path) -> pd.DataFrame:
    df = load_final_run_results(results_dir)
    if df.empty:
        return df

    rows = []
    for _, r in df.iterrows():
        site = 'outo' if 'outo' in r['test_dataset'] else 'tata'
        tgt  = r['target']       # 'uts' or 'ys'
        mk   = r['model_key']    # e.g. 'limix_ft'
        base = MODEL_KEY_TO_BASE.get(mk, mk)
        zs   = ZS.get((site, tgt, base))

        # A3/B3 ICL check — try log file first, then fall back to known values
        pool_tag = r['pool_tag']
        # The rs70 holdout dataset name differs by direction
        if pool_tag == 'open_tata':
            rs70_ds = f"outo_avg_ts_rs70" if tgt == 'uts' else "outo_avg_ys_rs70"
        else:  # open_outo
            rs70_ds = "tata_rm_rs70" if tgt == 'uts' else "tata_rp_rs70"

        a3_log = logdir / f"{'a3' if pool_tag == 'open_tata' else 'b3'}_{site}_{tgt}_{base}.log"
        a3_smape = parse_smape_from_log(a3_log) or A3_KNOWN.get((pool_tag, mk, rs70_ds))

        # ZS baseline on the same rs70 holdout (for A3/B3 delta)
        zs_rs70_log = logdir / f"zs_{site}_{tgt}_{base}.log"
        zs_rs70 = parse_smape_from_log(zs_rs70_log)
        a3_delta = round((a3_smape - zs_rs70) / zs_rs70 * 100, 1) if (a3_smape and zs_rs70) else None

        rows.append({
            'model':       mk,
            'base':        base,
            'direction':   'A (tata→outo)' if pool_tag == 'open_tata' else 'B (outo→tata)',
            'target':      tgt,
            'seed':        int(r['seed']),
            'smape_ft':    round(float(r['smape']), 4),
            'smape_zs':    zs,
            'ft_delta%':   round((float(r['smape']) - zs) / zs * 100, 1) if zs else None,
            'a3b3_smape':  round(a3_smape, 4) if a3_smape else None,
            'a3b3_zs_rs70': round(zs_rs70, 4) if zs_rs70 else None,
            'a3b3_delta%':  a3_delta,
        })

    return pd.DataFrame(rows)


def main():
    results_dir = ROOT / 'results' / 'finetune' / 'final'
    logdir = ROOT / 'results' / 'logs_ft_exp'

    tbl = build_table(results_dir, logdir)
    if tbl.empty:
        print("No results yet — run the final benchmark runs first.")
        return

    # Per-model mean over seeds
    agg_cols = {
        'smape_ft':       ('smape_ft', 'mean'),
        'smape_zs':       ('smape_zs', 'first'),
        'ft_delta%':      ('ft_delta%', 'mean'),
        'a3b3_smape':     ('a3b3_smape', 'first'),
        'a3b3_zs_rs70':   ('a3b3_zs_rs70', 'first'),
        'a3b3_delta%':    ('a3b3_delta%', 'first'),
        'n_seeds':        ('seed', 'count'),
    }
    agg = (
        tbl.groupby(['model', 'direction', 'target'])
           .agg(**agg_cols)
           .reset_index()
    )
    # Round aggregated means
    for col in ('smape_ft', 'ft_delta%'):
        agg[col] = agg[col].round(4)

    print("\n=== Final comparison table (mean over seeds) ===\n")
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(agg.to_string(index=False))

    # Also print a clean markdown-style summary
    print("\n=== Markdown summary ===\n")
    print(f"{'Model':<22} {'Dir':<16} {'Tgt':<5} {'FT SMAPE':>10} {'ZS SMAPE':>10} {'FT Δ%':>8} {'A3B3 SMAPE':>12} {'ZS rs70':>10} {'A3B3 Δ%':>10} {'Seeds':>6}")
    print("-" * 110)
    for _, row in agg.sort_values(['direction', 'target', 'model']).iterrows():
        a3b3_d = f"{row['a3b3_delta%']:+.1f}%" if pd.notna(row['a3b3_delta%']) else 'N/A'
        ft_d   = f"{row['ft_delta%']:+.1f}%"   if pd.notna(row['ft_delta%'])   else 'N/A'
        a3b3_s = f"{row['a3b3_smape']:.4f}"    if pd.notna(row['a3b3_smape']) else 'N/A'
        zs_r70 = f"{row['a3b3_zs_rs70']:.4f}"  if pd.notna(row['a3b3_zs_rs70']) else 'N/A'
        print(f"{row['model']:<22} {row['direction']:<16} {row['target']:<5} "
              f"{row['smape_ft']:>10.4f} {row['smape_zs']:>10.4f} {ft_d:>8} "
              f"{a3b3_s:>12} {zs_r70:>10} {a3b3_d:>10} {int(row['n_seeds']):>6}")

    out_csv = results_dir / 'final_comparison_table.csv'
    agg.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == '__main__':
    main()
