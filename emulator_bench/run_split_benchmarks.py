import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.rich_progress import progress  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    TASK_PARENTS,
    discover_split_jobs,
    task_dir,
)

TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def run_dir(output_root: Path, job: dict, seed: int) -> Path:
    return output_root / job["split_group"] / job["split_name"] / f"seed_{seed}"


def train_one(job: dict, seed: int, out_dir: Path, args, gpu: str | None) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_path", job["train_path"],
        "--val_path", job["val_path"],
        "--test_path", job["test_path"],
        "--out_dir", str(out_dir),
        "--device", "cuda:0" if gpu is not None else args.device,
        "--num_workers", str(args.num_workers),
        "--seed", str(seed),
    ]
    for flag in ("epochs", "max_rows"):
        if getattr(args, flag) is not None:
            cmd += [f"--{flag}", str(getattr(args, flag))]
    if args.amp:
        cmd.append("--amp")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main():
    parser = argparse.ArgumentParser(description="Run StructureFree-DTA with its original settings over EMULaToR split groups and seeds.")
    parser.add_argument("--task", choices=sorted(TASK_PARENTS), default="kd")
    parser.add_argument("--base_dir", default=None, help="Split root of the task. Default: the EMULaToR dataset folder of --task.")
    parser.add_argument("--output_root", default=None, help="Default: <EMULaToR>/outputs/baselines/StructureFree-DTA/<task>.")
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda:0", help="Device when --gpus is not set.")
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU ids, e.g. --gpus 0 1 2. Each run gets one GPU through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs per GPU.")
    parser.add_argument("--amp", action="store_true", help="bf16 on Ampere or newer, else fp16 with GradScaler. Default: the original fp16 autocast on CUDA.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test only. Default: original 150.")
    parser.add_argument("--max_rows", type=int, default=None, help="Smoke-test only: keep the first N rows of each split.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else task_dir(args.task)
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT / args.task
    jobs = discover_split_jobs(base_dir, args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {base_dir}")

    tasks = [(job, seed, run_dir(output_root, job, seed)) for job in jobs for seed in args.seeds]
    tasks = [task for task in tasks if args.overwrite or not (task[2] / "final_results_test.csv").exists()]
    slots = queue.Queue()
    for _ in range(args.runs_per_gpu):
        for gpu in args.gpus or [None]:
            slots.put(None if gpu is None else str(gpu))

    def work(task):
        gpu = slots.get()
        try:
            train_one(*task, args, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(slots.qsize()) as pool:
        for _ in progress(pool.map(work, tasks), total=len(tasks), desc="Benchmark runs", unit="run"):
            pass


if __name__ == "__main__":
    main()
