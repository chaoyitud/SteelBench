#!/usr/bin/env python3
"""
run_opensource_eval.py
Dispatch all open-source benchmark jobs across GPUs.

Datasets (28 folders total):
  steel_strength:  steel_ys_rs{50,60,70,80}  ×3 targets
  matbench_steels: matbench_ys_rs{50,60,70,80}
  nims_fatigue:    nims_{fs,uts,hv}_rs{50,60,70,80}

Models: same 13 as the private benchmark (see run_full_eval.py).
Total jobs: 28 datasets × 13 models = 364 jobs, 5 seeds each.

Usage:
    python run_opensource_eval.py --dry_run
    python run_opensource_eval.py --gpus 0,1,2,3 --seeds 5
"""
import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data" / "talent"
LOG_DIR     = PROJECT_DIR / "results" / "logs_open"
TEST_DIR    = PROJECT_DIR / "test"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"

# ── Dataset list ──────────────────────────────────────────────────────────────

STEEL_DATASETS = [
    f"steel_{target}_rs{frac}"
    for target in ["ys", "uts", "el"]
    for frac   in [50, 60, 70, 80]
]

MATBENCH_DATASETS = [
    f"matbench_ys_rs{frac}"
    for frac in [50, 60, 70, 80]
]

NIMS_DATASETS = [
    f"nims_{target}_rs{frac}"
    for target in ["fs", "uts", "hv"]
    for frac   in [50, 60, 70, 80]
]

ALL_DATASETS = STEEL_DATASETS + MATBENCH_DATASETS + NIMS_DATASETS  # 28 total

# ── Model flags (same as run_full_eval.py) ───────────────────────────────────

DEEP_MODELS = {
    # model: (normalization, cat_policy, num_policy)
    "limix":     ("none",     "indices",  "none"),
    "tabpfn_v2": ("none",     "indices",  "none"),
    "tabpfn_v3": ("none",     "indices",  "none"),
    "mitra":     ("none",     "indices",  "none"),
    "tabm":      ("quantile", "indices",  "none"),
    "ftt":       ("quantile", "indices",  "none"),
    "realmlp":   ("quantile", "indices",  "none"),
    "modernNCA": ("quantile", "tabr_ohe", "none"),
    "resnet":    ("standard", "ordinal",  "none"),
    "mlp":       ("standard", "ordinal",  "none"),
}

CLASSICAL_MODELS = {
    "catboost": "indices",
    "xgboost":  "ordinal",
    "lightgbm": "ordinal",
}


@dataclass
class Job:
    dataset:  str
    model:    str
    script:   str
    cmd_args: List[str]
    logfile:  Path
    gpu_id:   Optional[str] = None


def build_jobs(seeds: int, data_dir: Path, log_dir: Path,
               datasets: List[str]) -> List[Job]:
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    for dataset in datasets:
        # Skip if the TALENT folder doesn't exist yet
        dataset_path = data_dir / dataset
        if not dataset_path.exists():
            print(f"WARNING: Dataset folder not found: {dataset_path} — skipping jobs for this dataset")
            continue

        for model, (norm, cat, num) in DEEP_MODELS.items():
            lf  = log_dir / f"{dataset}__{model}.log"
            cmd = [
                str(VENV_PYTHON),
                str(TEST_DIR / "train_model_deep.py"),
                "--model_type",    model,
                "--dataset",       dataset,
                "--dataset_path",  str(data_dir),
                "--normalization", norm,
                "--cat_policy",    cat,
                "--num_policy",    num,
                "--seed_num",      str(seeds),
            ]
            jobs.append(Job(dataset, model, "deep", cmd, lf))

        for model, cat in CLASSICAL_MODELS.items():
            lf  = log_dir / f"{dataset}__{model}.log"
            cmd = [
                str(VENV_PYTHON),
                str(TEST_DIR / "train_model_classical.py"),
                "--model_type",   model,
                "--dataset",      dataset,
                "--dataset_path", str(data_dir),
                "--cat_policy",   cat,
                "--seed_num",     str(seeds),
            ]
            jobs.append(Job(dataset, model, "classical", cmd, lf))

    return jobs


def run_job(job: Job, gpu_id: str, dry_run: bool):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = job.cmd_args + ["--gpu", gpu_id]

    label = f"{job.dataset}__{job.model}"
    if dry_run:
        print(f"[DRY] {label}  gpu={gpu_id}")
        return job, True, 0.0

    # Skip already-completed jobs
    if job.logfile.exists() and "MAE MEAN" in job.logfile.read_text(errors='replace'):
        return job, True, 0.0

    t0 = time.time()
    with open(job.logfile, "w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=fh)
    elapsed = time.time() - t0
    return job, proc.returncode == 0, elapsed


def dispatch(jobs: List[Job], gpu_ids: List[str], dry_run: bool):
    total   = len(jobs)
    done    = [0]
    success = [0]

    def report(job, ok, elapsed):
        if dry_run:
            return
        done[0] += 1
        success[0] += ok
        tag = "OK  " if ok else "FAIL"
        print(f"[{tag}] ({done[0]:>3}/{total}) {job.dataset}__{job.model}"
              f"  gpu={job.gpu_id}  {elapsed:.0f}s  → {job.logfile.name}",
              flush=True)

    gpu_queues = {g: Queue() for g in gpu_ids}
    for i, job in enumerate(jobs):
        gpu = gpu_ids[i % len(gpu_ids)]
        job.gpu_id = gpu
        gpu_queues[gpu].put(job)

    def gpu_worker(gpu_id: str, q: Queue):
        while not q.empty():
            job = q.get()
            result = run_job(job, gpu_id, dry_run)
            report(*result)

    threads = [
        Thread(target=gpu_worker, args=(g, gpu_queues[g]), daemon=True)
        for g in gpu_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not dry_run:
        print(f"\n{'='*60}")
        print(f"Open-source eval complete: {success[0]}/{total} succeeded")
        print(f"Logs: {LOG_DIR}/")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated subset of datasets (default: all 28)")
    args = parser.parse_args()

    gpu_ids  = [g.strip() for g in args.gpus.split(",")]
    datasets = (args.datasets.split(",") if args.datasets
                else ALL_DATASETS)

    jobs = build_jobs(args.seeds, DATA_DIR, LOG_DIR, datasets)

    if not args.dry_run:
        print(f"Dispatching {len(jobs)} jobs across GPUs {gpu_ids} "
              f"(seeds={args.seeds})")
        print(f"Logs → {LOG_DIR}/\n")

    dispatch(jobs, gpu_ids, args.dry_run)


if __name__ == "__main__":
    main()
