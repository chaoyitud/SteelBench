#!/usr/bin/env python3
"""
run_full_eval.py
Dispatch all 208 benchmark runs across 4 GPUs.
Classical models (catboost, xgboost, lightgbm) run on GPU alongside deep models.

Usage:
    python run_full_eval.py [--dry_run] [--seeds 5] [--gpus 4,5,6,7]
"""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data" / "talent"
LOG_DIR     = PROJECT_DIR / "results" / "logs_full"
TEST_DIR    = PROJECT_DIR / "test"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"

TATA_DATASETS = [
    f"tata_{target}_rs{frac}"
    for target in ["rm", "rp"]
    for frac   in [50, 60, 70, 80]
]
OUTO_DATASETS = [
    f"outo_avg_{target}_rs{frac}"
    for target in ["ts", "ys"]
    for frac   in [50, 60, 70, 80]
]
ALL_DATASETS = TATA_DATASETS + OUTO_DATASETS

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
    # model: cat_policy
    "catboost": "indices",
    "xgboost":  "ordinal",
    "lightgbm": "ordinal",
}


@dataclass
class Job:
    dataset:  str
    model:    str
    script:   str           # "deep" or "classical"
    cmd_args: List[str]
    logfile:  Path
    gpu_id:   Optional[str] = None   # None = CPU


def build_jobs(seeds: int, data_dir: Path, log_dir: Path) -> List[Job]:
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    for dataset in ALL_DATASETS:
        for model, (norm, cat, num) in DEEP_MODELS.items():
            lf = log_dir / f"{dataset}__{model}.log"
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
            lf = log_dir / f"{dataset}__{model}.log"
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


def run_job(job: Job, gpu_id: Optional[str], dry_run: bool) -> tuple:
    """Run one job, return (job, success, elapsed_s)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = job.cmd_args + ["--gpu", gpu_id]
    else:
        # Pass 'cpu' so catboost uses task_type='CPU'; other classicals ignore it
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = job.cmd_args + ["--gpu", "cpu"]

    label = f"{job.dataset}__{job.model}"
    if dry_run:
        print(f"[DRY] {label}  gpu={gpu_id}")
        return job, True, 0.0

    if job.logfile.exists() and "MAE MEAN" in job.logfile.read_text():
        # Skip already-completed job (log contains final results)
        return job, True, 0.0

    t0 = time.time()
    with open(job.logfile, "w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=fh)
    elapsed = time.time() - t0
    return job, proc.returncode == 0, elapsed


def dispatch(jobs: List[Job], gpu_ids: List[str], dry_run: bool):
    """
    Dispatch all jobs across GPU queues (one active job per GPU).
    Deep and classical (catboost/xgboost/lightgbm) all run on GPU.
    """
    total   = len(jobs)
    done    = [0]
    success = [0]

    def report(job, ok, elapsed):
        if dry_run:
            return   # [DRY] already printed inside run_job
        done[0] += 1
        if ok:
            success[0] += 1
            tag = "OK  "
        else:
            tag = "FAIL"
        print(f"[{tag}] ({done[0]:>3}/{total}) {job.dataset}__{job.model}"
              f"  gpu={job.gpu_id}  {elapsed:.0f}s  → {job.logfile.name}",
              flush=True)

    # ── GPU workers (one thread per GPU, serial within each GPU) ─────────────
    gpu_queues = {g: Queue() for g in gpu_ids}

    # Round-robin assignment for all jobs
    for i, job in enumerate(jobs):
        gpu = gpu_ids[i % len(gpu_ids)]
        job.gpu_id = gpu
        gpu_queues[gpu].put(job)

    def gpu_worker(gpu_id: str, q: Queue):
        while not q.empty():
            job = q.get()
            j, ok, elapsed = run_job(job, gpu_id, dry_run)
            report(j, ok, elapsed)

    gpu_threads = [
        Thread(target=gpu_worker, args=(g, gpu_queues[g]), daemon=True)
        for g in gpu_ids
    ]
    for t in gpu_threads:
        t.start()

    for t in gpu_threads:
        t.join()

    if not dry_run:
        print(f"\n{'='*60}")
        print(f"Full eval complete: {success[0]}/{total} succeeded")
        print(f"Logs: {LOG_DIR}/")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip jobs whose log file already exists")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",")]
    jobs    = build_jobs(args.seeds, DATA_DIR, LOG_DIR)

    if not args.dry_run:
        print(f"Dispatching {len(jobs)} jobs across GPUs {gpu_ids} "
              f"(seeds={args.seeds})")
        print(f"Logs → {LOG_DIR}/\n")

    dispatch(jobs, gpu_ids, args.dry_run)


if __name__ == "__main__":
    main()
