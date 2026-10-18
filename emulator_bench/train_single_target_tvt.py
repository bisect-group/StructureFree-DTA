import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.rich_progress import progress  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import append_csv_row, read_table, regression_metrics, resolve_amp_dtype, save_json, set_seed  # noqa: E402
from models import AffinityPredictor, DrugTargetInteractionLoss  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Collate:
    """Tokenizes a batch like the original `collate_fn`: pad to the fixed maximum length."""

    def __init__(self, args):
        self.molecule_tokenizer = AutoTokenizer.from_pretrained(args.molecule_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_model_name)
        self.max_molecule_length = args.max_molecule_length
        self.max_protein_length = args.max_protein_length

    def __call__(self, rows):
        molecules, proteins, labels = zip(*rows)
        mol = self.molecule_tokenizer(list(molecules), padding="max_length", truncation=True, max_length=self.max_molecule_length, return_tensors="pt")
        prot = self.protein_tokenizer(list(proteins), padding="max_length", truncation=True, max_length=self.max_protein_length, return_tensors="pt")
        return {
            "molecule_input_ids": mol.input_ids,
            "molecule_attention_mask": mol.attention_mask,
            "protein_input_ids": prot.input_ids,
            "protein_attention_mask": prot.attention_mask,
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def make_rows(frame: pd.DataFrame, args):
    return list(zip(frame[args.smiles_col].astype(str), frame[args.sequence_col].astype(str), frame[args.target_col].astype(float)))


def make_loader(rows, batch_size, args, collate, shuffle):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, amp_dtype, device, desc):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()
    for batch in progress(loader, desc=desc, unit="batch", leave=False):
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = loss_fn(model(batch).view(-1), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, desc="Eval"):
    model.eval()
    total_loss, total_samples = 0.0, 0
    preds, truths = [], []
    for batch in progress(loader, desc=desc, unit="batch", leave=False):
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        output = model(batch).view(-1)
        total_loss += loss_fn(output, labels).item() * labels.size(0)
        total_samples += labels.size(0)
        preds.append(output.cpu())
        truths.append(labels.cpu())
    y_true, y_pred = torch.cat(truths).numpy(), torch.cat(preds).numpy()
    return y_true, y_pred, total_loss / total_samples


def write_split_results(out_dir, name, frame, y_true, y_pred, args):
    out = frame[[args.sequence_col, args.smiles_col, args.target_col]].copy()
    out["y_true"], out["y_pred"] = y_true, y_pred
    out.to_csv(out_dir / f"pred_label_{name}.csv", index=False)
    metrics = regression_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(out_dir / f"final_results_{name}.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train StructureFree-DTA end to end on explicit train/val/test files with the original settings.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sequence_col", default="sequence")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--target_col", default="log10_value")
    parser.add_argument("--protein_model_name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--molecule_model_name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--max_protein_length", type=int, default=128)
    parser.add_argument("--max_molecule_length", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 768, 512, 256, 1])
    parser.add_argument("--inception_out_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for the final val/test predictions only.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on Ampere or newer, fp16 with GradScaler on older GPUs. Default: fp16 autocast with GradScaler on CUDA, as in the original.")
    parser.add_argument("--max_rows", type=int, default=None, help="Smoke-test only: keep the first N rows of each split.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {name: read_table(Path(getattr(args, f"{name}_path"))) for name in ("train", "val", "test")}
    if args.max_rows:
        frames = {name: frame.head(args.max_rows).reset_index(drop=True) for name, frame in frames.items()}
    rows = {name: make_rows(frame, args) for name, frame in frames.items()}
    collate = Collate(args)

    model = AffinityPredictor(
        protein_model_name=args.protein_model_name,
        molecule_model_name=args.molecule_model_name,
        hidden_sizes=args.hidden_sizes,
        inception_out_channels=args.inception_out_channels,
        dropout=args.dropout,
    ).to(device)
    loss_fn = DrugTargetInteractionLoss(alpha=args.loss_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    if args.amp:
        amp_dtype = resolve_amp_dtype(device)[0]
    else:
        amp_dtype = torch.float16 if device.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    train_loader = make_loader(rows["train"], args.batch_size, args, collate, shuffle=True)
    val_loader = make_loader(rows["val"], args.batch_size, args, collate, shuffle=False)
    best_path = out_dir / "bestmodel.pth"
    best_val_loss, best_epoch, patience_counter = float("inf"), 0, 0
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()
    (out_dir / "logfile.csv").unlink(missing_ok=True)
    best_path.unlink(missing_ok=True)
    epoch = 0

    for epoch in progress(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, amp_dtype, device, f"Epoch {epoch} train")
        scheduler.step()
        val_true, val_pred, val_loss = evaluate(model, val_loader, loss_fn, device, f"Epoch {epoch} val")
        val_metrics = regression_metrics(val_true, val_pred)
        append_csv_row(
            out_dir / "logfile.csv",
            {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss, "val_loss": val_loss,
             **{f"val_{key}": value for key, value in val_metrics.items()}, "elapsed_seconds": time.time() - started},
        )
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics = {}
    for name in ("val", "test"):
        loader = make_loader(rows[name], args.eval_batch_size, args, collate, shuffle=False)
        y_true, y_pred, _ = evaluate(model, loader, loss_fn, device, f"Final {name}")
        final_metrics[name] = write_split_results(out_dir, name, frames[name], y_true, y_pred, args)

    save_json(
        out_dir / "run_summary.json",
        {
            "started_at": started_at,
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "epochs_run": epoch,
            "final_val_metrics": final_metrics["val"],
            "final_test_metrics": final_metrics["test"],
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
