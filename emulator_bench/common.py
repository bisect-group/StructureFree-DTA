import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT.parents[1]
DATASETS_ROOT = EMULATOR_ROOT / "data" / "processed" / "datasets"
DEFAULT_OUTPUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "StructureFree-DTA"
TASK_PARENTS = {
    "kcat": "kinetic_dataset",
    "km": "kinetic_dataset",
    "ki": "kinetic_dataset",
    "ic50": "binding_affinity_dataset",
    "kd": "binding_affinity_dataset",
    "ec50": "binding_affinity_dataset",
}
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
DEFAULT_SEEDS = [666, 777, 888]


def task_dir(task: str) -> Path:
    return DATASETS_ROOT / TASK_PARENTS[task] / task


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_csv_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def discover_split_jobs(base_dir: Path, split_groups: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """Find `<group>/{train,val,test}.parquet` or `<group>/threshold_*/{train,val,test}.parquet` jobs."""
    jobs: List[Dict[str, str]] = []
    for split_group in split_groups or DEFAULT_SPLIT_GROUPS:
        group_dir = Path(base_dir) / split_group
        candidates = [(split_group.removeprefix("random_splits_grouped_").replace("_splits", ""), group_dir)]
        if not (group_dir / "train.parquet").exists():
            candidates = [(child.name, child) for child in sorted(group_dir.glob("threshold_*")) if child.is_dir()]
        for split_name, root in candidates:
            if all((root / f"{stem}.parquet").exists() for stem in ("train", "val", "test")):
                jobs.append(
                    {
                        "split_group": split_group,
                        "split_name": split_name,
                        **{f"{stem}_path": str(root / f"{stem}.parquet") for stem in ("train", "val", "test")},
                    }
                )
    return jobs


def concordance_index(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return 0.5
    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]
    ranks = {value: idx + 1 for idx, value in enumerate(sorted(set(yp)))}
    bit = np.zeros(len(ranks) + 2, dtype=np.int64)

    def add(index: int) -> None:
        while index < len(bit):
            bit[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(bit[index])
            index -= index & -index
        return total

    permissible = 0
    concordant = 0
    previous = 0
    start = 0
    while start < len(yt):
        end = start + 1
        while end < len(yt) and yt[end] == yt[start]:
            end += 1
        for pred in yp[start:end]:
            concordant += prefix(ranks[pred] - 1)
            permissible += previous
        for pred in yp[start:end]:
            add(ranks[pred])
            previous += 1
        start = end
    return float(concordant / permissible) if permissible else 0.5


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    from scipy import stats

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    constant = np.std(y_true) == 0 or np.std(y_pred) == 0
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2_score": 1.0 - float(np.sum(np.square(residual))) / ss_tot if ss_tot > 0 else 0.0,
        "pearson": 0.0 if constant else float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman": 0.0 if constant else float(stats.spearmanr(y_true, y_pred).statistic),
        "ci": concordance_index(y_true, y_pred),
    }


def summarize_seed_runs(rows: List[Dict], group_cols: List[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out_rows = []
    for keys, group in pd.DataFrame(rows).groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_var"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)
