#!/usr/bin/env python3
"""
run_classical_eval.py
Run all classical benchmark jobs (catboost / xgboost / lightgbm) across 4 GPUs.
Each job writes its own log file under results/logs_full/.
Progress is printed to stdout so you can `tail -f` or read naturally.

Usage:
    python run_classical_eval.py [--seeds 5] [--gpus 4,5,6,7] [--dry_run]
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

PROJECT_DIR  = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_DIR / "data" / "talent"
LOG_DIR      = PROJECT_DIR / "results" / "logs_full"
VENV_PYTHON  = PROJECT_DIR / ".venv" / "bin" / "python"
TRAIN_SCRIPT = PROJECT_DIR / "test" / "train_model_classical.py"

DATASETS = [
    f"outo_avg_ts_rs{s}" for s in [50, 60, 70, 80]
] + [
    f"outo_avg_ys_rs{s}" for s in [50, 60, 70, 80]
] + [
    f"tata_rm_rs{s}" for s in [50, 60, 70, 80]
] + [
    f"tata_rp_rs{s}" for s in [50, 60, 70, 80]
]

# model → cat_policy
MODELS = {
    "catboost": "indices",
    "xgboost":  "ordinal",
    "lightgbm": "ordinal",
}

SUCCESS_MARKER = "MAE MEAN"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def build_jobs(seeds: int) -> list[dict]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in DATASETS:
        for model, cat_policy in MODELS.items():
            logfile = LOG_DIR / f"{dataset}__{model}.log"
            cmd = [
                str(VENV_PYTHON),
                str(TRAIN_SCRIPT),
                "--model_type",   model,
                "--dataset",      dataset,
                "--dataset_path", str(DATA_DIR),
                "--cat_policy",   cat_policy,
                "--seed_num",     str(seeds),
            ]
            jobs.append({"dataset": dataset, "model": model,
                         "logfile": logfile, "cmd": cmd})
    return jobs


def run_one(job: dict, gpu_id: str, dry_run: bool) -> bool:
    """Run a single job on the given GPU. Returns True on success."""
    label   = f"{job['dataset']}__{job['model']}"
    logfile = job["logfile"]

    # Skip if already complete
    if logfile.exists() and SUCCESS_MARKER in logfile.read_text(errors="replace"):
        print(f"[{ts()}] SKIP  {label}  (already complete)", flush=True)
        return True

    cmd = job["cmd"] + ["--gpu", gpu_id]
    env = os.environ.copy()
    env["PYTHONPATH"]        = str(PROJECT_DIR)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    if dry_run:
        print(f"[{ts()}] DRY   GPU={gpu_id}  {label}", flush=True)
        return True

    print(f"[{ts()}] START GPU={gpu_id}  {label}", flush=True)
    t0 = time.time()
    with open(logfile, "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=fh)
    elapsed = time.time() - t0

    ok = result.returncode == 0 and (
        logfile.exists() and SUCCESS_MARKER in logfile.read_text(errors="replace")
    )
    status = "OK   " if ok else "FAIL "
    print(f"[{ts()}] {status} GPU={gpu_id}  {label}  ({elapsed:.0f}s)", flush=True)
    return ok


def gpu_worker(gpu_id: str, q: Queue, dry_run: bool, results: list):
    while not q.empty():
        try:
            job = q.get_nowait()
        except Exception:
            break
        ok = run_one(job, gpu_id, dry_run)
        results.append((f"{job['dataset']}__{job['model']}", ok))
        q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, default=5)
    parser.add_argument("--gpus",    type=str, default="4,5,6,7")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gpu_ids = args.gpus.split(",")
    jobs    = build_jobs(args.seeds)

    # Report
    skip = sum(
        1 for j in jobs
        if j["logfile"].exists()
        and SUCCESS_MARKER in j["logfile"].read_text(errors="replace")
    )
    todo = len(jobs) - skip
    print(f"[{ts()}] Classical eval: {len(jobs)} total, {skip} already done, {todo} to run")
    print(f"[{ts()}] GPUs: {gpu_ids}   Seeds per job: {args.seeds}")
    print()

    # Fill a shared queue
    q: Queue = Queue()
    for j in jobs:
        q.put(j)

    results: list = []

    # One thread per GPU, each drains from the shared queue
    threads = []
    for gpu in gpu_ids:
        t = Thread(target=gpu_worker, args=(gpu, q, args.dry_run, results), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Summary
    n_ok   = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print()
    print(f"[{ts()}] Done: {n_ok} OK, {n_fail} FAILED")
    if n_fail:
        print("Failed jobs:")
        for label, ok in results:
            if not ok:
                print(f"  {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
