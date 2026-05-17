#!/usr/bin/env python3
"""
collect_results.py
Parse TALENT log files and write a single CSV summary.

Usage:
    python collect_results.py [--log_dir results/logs/] [--output results/smoke_test_results.csv]
    python collect_results.py --log_dir results/logs_full/ --output results/full_results.csv
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

# ── Regex patterns ──────────────────────────────────────────────────────────
RE_MEAN = re.compile(
    r'^(\w+)\s+MEAN\s*=\s*([\d.eE+\-]+)\s*[±\+\-]+\s*([\d.eE+\-]+)',
    re.IGNORECASE,
)
RE_RESULTS = re.compile(
    r'^(\w+)\s+Results:\s*(.+)',
    re.IGNORECASE,
)
RE_TRIALS = re.compile(r'^(\S+):\s+(\d+)\s+Trials', re.IGNORECASE)

# Canonical output metrics (in preferred column order)
METRICS = ['MAE', 'R2', 'RMSE', 'SMAPE']
TIME_ALIASES = {'Total_Time', 'Time', 'Fit_Time', 'Inference_Time'}

CSV_COLUMNS = [
    'dataset', 'model', 'status', 'seed_num',
    'MAE_mean', 'MAE_std',
    'R2_mean', 'R2_std',
    'RMSE_mean', 'RMSE_std',
    'SMAPE_mean', 'SMAPE_std',
    'Time_mean', 'Time_std',
    'log_file',
]


def parse_log(log_path: Path) -> dict:
    """Parse a single TALENT log file and return a result dict."""
    text = log_path.read_text(errors='replace')
    lines = text.splitlines()

    dataset, model = log_path.stem.split('__', 1)

    row = {
        'dataset': dataset,
        'model': model,
        'status': 'OK',
        'seed_num': None,
        'log_file': str(log_path),
    }
    for m in METRICS:
        row[f'{m}_mean'] = float('nan')
        row[f'{m}_std']  = float('nan')
    row['Time_mean'] = float('nan')
    row['Time_std']  = float('nan')

    # Detect skipped or failed
    if 'SKIPPED' in text:
        row['status'] = 'SKIPPED'
        return row

    has_mean = False
    for line in lines:
        line = line.strip()

        # Detect seed count
        m = RE_TRIALS.match(line)
        if m:
            row['seed_num'] = int(m.group(2))
            continue

        # Detect MEAN lines
        m = RE_MEAN.match(line)
        if m:
            metric_raw = m.group(1)
            mean_val   = float(m.group(2))
            std_val    = float(m.group(3))
            has_mean   = True

            # Map metric name to column
            mu = metric_raw.upper()
            if mu in {mm.upper() for mm in METRICS}:
                # Find canonical name (case-insensitive match)
                canonical = next(mm for mm in METRICS if mm.upper() == mu)
                row[f'{canonical}_mean'] = mean_val
                row[f'{canonical}_std']  = std_val
            elif metric_raw in {'Total_Time', 'Time'}:
                row['Time_mean'] = mean_val
                row['Time_std']  = std_val

    # Mark as failed if no MEAN lines found and there are error indicators
    if not has_mean:
        if any(kw in text for kw in ('Traceback', 'Error', 'FAILED', 'assert')):
            row['status'] = 'FAILED'
        else:
            row['status'] = 'FAILED'   # no output at all → also failed

    return row


def fmt(val) -> str:
    """Format a float for the summary table."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 'NaN'
    return f'{val:.4f}'


def main():
    parser = argparse.ArgumentParser(description='Collect TALENT smoke test results into CSV')
    parser.add_argument('--log_dir', default='results/logs/',
                        help='Directory containing .log files (default: results/logs/)')
    parser.add_argument('--output', default='results/smoke_test_results.csv',
                        help='Output CSV path (default: results/smoke_test_results.csv)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output  = Path(args.output)

    if not log_dir.exists():
        print(f'ERROR: log directory not found: {log_dir}', file=sys.stderr)
        sys.exit(1)

    log_files = sorted(f for f in log_dir.glob('*.log') if '__' in f.stem)
    if not log_files:
        print(f'WARNING: no .log files found in {log_dir}', file=sys.stderr)
        sys.exit(0)

    rows = [parse_log(f) for f in log_files]

    # Write CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # ── Dataset decomposition → full_results_parsed.csv ─────────────────────
    def _source(d: str) -> str:
        return "Tata" if d.startswith("tata") else "Outo"

    def _target(d: str) -> str:
        # tata_rm_rs70   → RM
        # outo_avg_ts_rs70 → AVG_TS
        return d.split("_rs")[0].split("_", 1)[1].upper()

    def _train_pct(d: str) -> int:
        return int(d.split("_rs")[1])

    MODEL_FAMILY = {
        m: "TFM"
        for m in {"limix", "tabpfn_v2", "tabpfn_v3", "mitra"}
    }
    MODEL_FAMILY.update({
        m: "Deep"
        for m in {"tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp"}
    })
    MODEL_FAMILY.update({
        m: "Classical"
        for m in {"catboost", "xgboost", "lightgbm", "RandomForest"}
    })

    parsed_columns = CSV_COLUMNS + ['source', 'target', 'train_pct', 'model_family']
    parsed_rows = []
    for row in rows:
        d = row['dataset']
        pr = dict(row)
        try:
            pr['source']       = _source(d)
            pr['target']       = _target(d)
            pr['train_pct']    = _train_pct(d)
            pr['model_family'] = MODEL_FAMILY.get(row['model'], 'Other')
        except Exception:
            pr['source'] = pr['target'] = pr['model_family'] = ''
            pr['train_pct'] = None
        parsed_rows.append(pr)

    # Derive parsed output path from output path
    p = output.with_name(output.stem + '_parsed' + output.suffix)
    with open(p, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=parsed_columns)
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow(row)

    print(f'Parsed results written to: {p}')

    # Print summary table
    col_w = {'dataset': 20, 'model': 18, 'status': 8,
             'MAE_mean': 11, 'SMAPE_mean': 11}
    header = (f"{'dataset':<{col_w['dataset']}} "
              f"{'model':<{col_w['model']}} "
              f"{'status':<{col_w['status']}} "
              f"{'MAE_mean':>{col_w['MAE_mean']}} "
              f"{'SMAPE_mean':>{col_w['SMAPE_mean']}}")
    sep = '=' * (sum(col_w.values()) + len(col_w))

    print(f'\n{sep}')
    print(f'Results summary  ({output})')
    print(sep)
    print(header)
    print('-' * (sum(col_w.values()) + len(col_w)))
    for row in rows:
        print(f"{row['dataset']:<{col_w['dataset']}} "
              f"{row['model']:<{col_w['model']}} "
              f"{row['status']:<{col_w['status']}} "
              f"{fmt(row['MAE_mean']):>{col_w['MAE_mean']}} "
              f"{fmt(row['SMAPE_mean']):>{col_w['SMAPE_mean']}}")

    n_ok      = sum(1 for r in rows if r['status'] == 'OK')
    n_failed  = sum(1 for r in rows if r['status'] == 'FAILED')
    n_skipped = sum(1 for r in rows if r['status'] == 'SKIPPED')
    print(sep)
    print(f'Total: {len(rows)} runs | OK: {n_ok} | FAILED: {n_failed} | SKIPPED: {n_skipped}')
    print(f'Full results written to: {output}\n')


if __name__ == '__main__':
    main()
