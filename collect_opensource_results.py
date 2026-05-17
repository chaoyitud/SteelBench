#!/usr/bin/env python3
"""
collect_opensource_results.py
Parse open-source experiment logs and produce:
  - results/opensource_results_parsed.csv  (open-source rows only)
  - results/all_results_merged.csv         (private + open combined, tier column)

Usage:
    python collect_opensource_results.py \
        --log_dir     results/logs_open/ \
        --output      results/opensource_results_parsed.csv \
        --private_csv results/full_results_parsed.csv \
        --merged_csv  results/all_results_merged.csv
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

# ── Regex (identical to collect_results.py) ───────────────────────────────────
RE_MEAN = re.compile(
    r'^(\w+)\s+MEAN\s*=\s*([\d.eE+\-]+)\s*[±\+\-]+\s*([\d.eE+\-]+)',
    re.IGNORECASE,
)
RE_TRIALS = re.compile(r'^(\S+):\s+(\d+)\s+Trials', re.IGNORECASE)

METRICS = ['MAE', 'R2', 'RMSE', 'SMAPE']

CSV_COLUMNS = [
    'dataset', 'model', 'status', 'seed_num',
    'MAE_mean', 'MAE_std',
    'R2_mean',  'R2_std',
    'RMSE_mean', 'RMSE_std',
    'SMAPE_mean', 'SMAPE_std',
    'Time_mean', 'Time_std',
    'log_file',
]

PARSED_COLUMNS = CSV_COLUMNS + [
    'source', 'target', 'target_display', 'train_pct', 'model_family', 'tier',
]

MODEL_FAMILY = {
    **{m: "TFM"       for m in {"limix", "tabpfn_v2", "tabpfn_v3", "mitra"}},
    **{m: "Deep"      for m in {"tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp"}},
    **{m: "Classical" for m in {"catboost", "xgboost", "lightgbm"}},
}

TARGET_DISPLAY = {
    'YS':  'Yield Strength',
    'UTS': 'Tensile Strength',
    'EL':  'Elongation',
    'FS':  'Fatigue Strength',
    'HV':  'Vickers Hardness',
}


# ── Log parser ────────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> dict:
    text  = log_path.read_text(errors='replace')
    lines = text.splitlines()

    dataset, model = log_path.stem.split('__', 1)
    row = {
        'dataset':   dataset,
        'model':     model,
        'status':    'OK',
        'seed_num':  None,
        'log_file':  str(log_path),
    }
    for m in METRICS:
        row[f'{m}_mean'] = float('nan')
        row[f'{m}_std']  = float('nan')
    row['Time_mean'] = float('nan')
    row['Time_std']  = float('nan')

    if 'SKIPPED' in text:
        row['status'] = 'SKIPPED'
        return row

    has_mean = False
    for line in lines:
        line = line.strip()
        m = RE_TRIALS.match(line)
        if m:
            row['seed_num'] = int(m.group(2))
            continue
        m = RE_MEAN.match(line)
        if m:
            metric_raw = m.group(1).upper()
            mean_val   = float(m.group(2))
            std_val    = float(m.group(3))
            has_mean   = True
            if metric_raw in {mm.upper() for mm in METRICS}:
                canonical = next(mm for mm in METRICS if mm.upper() == metric_raw)
                row[f'{canonical}_mean'] = mean_val
                row[f'{canonical}_std']  = std_val
            elif metric_raw in {'TOTAL_TIME', 'TIME', 'FIT_TIME', 'INFERENCE_TIME'}:
                row['Time_mean'] = mean_val
                row['Time_std']  = std_val

    if not has_mean:
        row['status'] = 'FAILED'
    return row


# ── Dataset name decoder ──────────────────────────────────────────────────────

def parse_folder_name(dataset_col: str) -> tuple:
    """
    steel_ys_rs70     → ('steel_strength', 'YS',  70, 'open')
    matbench_ys_rs60  → ('matbench_steels','YS',  60, 'open')
    nims_fs_rs80      → ('nims_fatigue',   'FS',  80, 'open')
    """
    d = dataset_col
    try:
        if d.startswith('steel_'):
            parts  = d.split('_')          # ['steel','ys','rs70']
            source = 'steel_strength'
            target = parts[1].upper()
            pct    = int(parts[2][2:])
        elif d.startswith('matbench_'):
            parts  = d.split('_')          # ['matbench','ys','rs60']
            source = 'matbench_steels'
            target = parts[1].upper()
            pct    = int(parts[2][2:])
        elif d.startswith('nims_'):
            parts  = d.split('_')          # ['nims','fs','rs80']
            source = 'nims_fatigue'
            target = parts[1].upper()
            pct    = int(parts[2][2:])
        else:
            return ('unknown', d.upper(), 0, 'open')
        return (source, target, pct, 'open')
    except Exception:
        return ('unknown', d, 0, 'open')


# ── CSV writing ───────────────────────────────────────────────────────────────

def fmt(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 'NaN'
    return f'{val:.4f}'


def enrich(rows: list) -> list:
    enriched = []
    for row in rows:
        pr = dict(row)
        source, target, pct, tier = parse_folder_name(row['dataset'])
        pr['source']         = source
        pr['target']         = target
        pr['target_display'] = TARGET_DISPLAY.get(target, target)
        pr['train_pct']      = pct
        pr['model_family']   = MODEL_FAMILY.get(row['model'], 'Other')
        pr['tier']           = tier
        enriched.append(pr)
    return enriched


def write_csv(rows: list, path: Path, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_dir',     default='results/logs_open/')
    parser.add_argument('--output',      default='results/opensource_results_parsed.csv')
    parser.add_argument('--private_csv', default='results/full_results_parsed.csv')
    parser.add_argument('--merged_csv',  default='results/all_results_merged.csv')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f'ERROR: log directory not found: {log_dir}', file=sys.stderr)
        sys.exit(1)

    log_files = sorted(f for f in log_dir.glob('*.log') if '__' in f.stem)
    if not log_files:
        print(f'WARNING: no .log files in {log_dir}')
        sys.exit(0)

    rows = [parse_log(f) for f in log_files]

    # ── Open-source CSV ───────────────────────────────────────────────────────
    enriched = enrich(rows)
    output   = Path(args.output)
    write_csv(enriched, output, PARSED_COLUMNS)
    print(f'Open-source results written to: {output}  ({len(enriched)} rows)')

    n_ok     = sum(1 for r in enriched if r['status'] == 'OK')
    n_failed = sum(1 for r in enriched if r['status'] == 'FAILED')
    print(f'  OK: {n_ok}  FAILED: {n_failed}  '
          f'SKIPPED: {len(enriched) - n_ok - n_failed}')

    # ── Merged CSV ────────────────────────────────────────────────────────────
    private_csv = Path(args.private_csv)
    merged_csv  = Path(args.merged_csv)

    if not private_csv.exists():
        print(f'WARNING: private CSV not found: {private_csv} — merged CSV will contain open rows only')
        merged_rows = enriched
    else:
        import csv as _csv
        with open(private_csv) as f:
            private_rows = list(_csv.DictReader(f))

        # Add tier=private if missing
        for pr in private_rows:
            pr.setdefault('tier', 'private')
            pr.setdefault('target_display', pr.get('target', ''))

        merged_rows = private_rows + enriched

    # Unified column set (superset of both)
    merged_cols = list(dict.fromkeys(
        list(PARSED_COLUMNS) +
        [c for c in (private_rows[0].keys() if private_csv.exists() and private_rows else [])
         if c not in PARSED_COLUMNS]
    ))
    write_csv(merged_rows, merged_csv, merged_cols)
    n_priv = sum(1 for r in merged_rows if r.get('tier') == 'private')
    n_open = sum(1 for r in merged_rows if r.get('tier') == 'open')
    print(f'\nMerged CSV written to: {merged_csv}')
    print(f'  private rows: {n_priv}  open rows: {n_open}  total: {len(merged_rows)}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'dataset':<28} {'model':<14} {'status':<8} {'MAE_mean':>10} {'SMAPE_mean':>10}")
    print('-' * 76)
    for row in enriched:
        print(f"{row['dataset']:<28} {row['model']:<14} {row['status']:<8} "
              f"{fmt(row['MAE_mean']):>10} {fmt(row['SMAPE_mean']):>10}")


if __name__ == '__main__':
    main()
