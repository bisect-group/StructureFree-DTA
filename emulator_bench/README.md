# emulator_bench

Multi-seed evaluation of StructureFree-DTA with its original published settings on the default EMULaToR splits.

The model is the original `src/` code: ESM2-t6-8M and ChemBERTa-77M-MLM encoders (both fine-tuned), two Residual Inception blocks, and a 4-layer Mish MLP. Training follows `cfg/config.yaml` and `src/main.py`. No hyperparameter is tuned.

## Data

`--task` selects the dataset folder under `data/processed/datasets`:

- `kcat`, `km`, `ki`: `kinetic_dataset/<task>`
- `ic50`, `kd`, `ec50`: `binding_affinity_dataset/<task>`

The paper uses Kd, so `kd` is the default. Use `--base_dir` for another split root.

Split groups run by default: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`. Missing groups are skipped. Each folder holds `train/val/test.parquet` with `sequence`, `smiles`, `log10_value`.

## Scripts

- `train_single_target_tvt.py`: trains one split and seed. Writes `bestmodel.pth`, `logfile.csv`, `pred_label_{val,test}.csv`, `final_results_{val,test}.csv`, `run_summary.json`.
- `run_split_benchmarks.py`: runs all discovered splits and seeds (default 666 777 888). Skips finished runs. `--gpus 0 1 2` gives each run one GPU through `CUDA_VISIBLE_DEVICES`. `--runs_per_gpu N` sets the runs per GPU. Without `--gpus`, runs go one at a time on `--device`. `--amp` switches the training precision from the original fp16 autocast to bf16 (Ampere or newer; fp16 with GradScaler on older GPUs). Run folders are separate, so parallel runs never overwrite each other.
- `aggregate_results.py`: prints mean and std over seeds from an output root.

Outputs go to `--output_root`, default `outputs/baselines/StructureFree-DTA/<task>`, as `<split_group>/<split_name>/seed_<seed>`. Split groups without `threshold_*` folders use `all` as `<split_name>`. Split files are never modified.

## Settings

See `REPORT.md` for the settings table and the paper/code differences. `--epochs` and `--max_rows` are for smoke tests only.

See `commands.txt` for run commands.
