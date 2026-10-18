# StructureFree-DTA emulator_bench report

## Method
Structure-free DTA model (Bidgoli et al., J. Cheminformatics 2026). Inputs are a FASTA sequence and a SMILES string.
ESM2-t6-8M encodes the protein. ChemBERTa-77M-MLM encodes the molecule. Both encoders are fine-tuned end to end (gradient checkpointing on).
Each encoder output is mean pooled over all token positions, padding included. The two vectors are concatenated (320 + 384 = 704).
Two Residual Inception blocks (Conv1d kernels 1 and 3, BatchNorm, ReLU, dropout) mix the vector.
A 4-layer MLP with Mish (1024-768-512-256-1) gives one value.
Loss: alpha * (1 - cosine over the batch vector) + (1 - alpha) * summed MSE, alpha = 0.5.
The paper reports Davis, KIBA and BindingDB (Kd).

## Original default settings
Source for the code values: `cfg/config.yaml`, `src/main.py`, `src/trainer.py`, `src/models.py`, `src/config.py`.

| Setting | Value used | Source |
|---|---|---|
| Protein / molecule encoder | esm2_t6_8M_UR50D / ChemBERTa-77M-MLM, fine-tuned | config.yaml; paper p.4 |
| Max tokens protein / molecule | 128 / 128 | config.yaml data (paper p.4 says 1024 / 128) |
| Padding | `max_length` (fixed) | dataloader.py collate_fn |
| Pooling | plain mean over all positions | models.py forward |
| Inception out channels, dropout | 256 (unused by the code), 0.05 | config.yaml |
| MLP | 1024,768,512,256,1, Mish | config.yaml; paper p.5 |
| Optimiser | AdamW, lr 5e-5, wd 1e-5 | config.yaml; main.py |
| Scheduler | CosineAnnealingLR, T_max = epochs, eta_min 0, no warmup, per epoch | main.py (paper p.6-7 says 10% linear warm-up) |
| Batch size | 64, fixed | config.yaml (paper p.7 says 64 -> 512 -> 2048) |
| Epochs | 150 | config.yaml (paper Table 2 reports 100-110 epochs) |
| Early stopping | patience 10 on validation loss, best checkpoint by validation loss | config.yaml; trainer.py |
| Gradient clip | none | config.yaml `clip_grad_norm: null` |
| Precision | fp16 autocast + GradScaler for training, fp32 validation | trainer.py |
| Loss alpha | 0.5 | config.yaml |

Paper and code disagree on token length, warm-up, batch schedule and epochs. The wrapper follows the code.
The code also uses a 10% validation split of the training folds. The wrapper uses the EMULaToR val file instead.

## What the wrapper does now
- Reads `train/val/test.parquet` of each default split folder of a task (`--task`, default kd).
- Trains the original `AffinityPredictor` and `DrugTargetInteractionLoss` (imported from `src/`) end to end with the settings above.
- Epoch loop copies `Trainer`: train epoch, cosine step, validation loss per batch (batch size 64, no autocast), strict-improvement checkpoint, stop after 10 non-improving epochs.
- Reloads the best checkpoint and writes val and test predictions and metrics (RMSE, MAE, R2, Pearson, Spearman, CI).
- `run_split_benchmarks.py` loops splits x seeds (666 777 888) and skips finished runs.

## Files
Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `default_hparams.json` (tuned values, `best_hparams` key), `cache_embeddings.py`, `feature_pipeline.py`, `dataset.py`, `modeling.py` (frozen-embedding cache path).
Rewritten: `common.py` (paths, split discovery, metrics), `train_single_target_tvt.py`, `run_split_benchmarks.py`, `aggregate_results.py`, `README.md`, `commands.txt`.
Changed: `aggregate_results.py` (no `retrain_from_optuna`, no legacy `random_splits` alias, no train/size scan), `requirements.txt` (optuna line removed; original code untouched).

## Bugs fixed
- The old wrapper froze the encoders and used cached, mask-aware mean-pooled embeddings. This changed the model (the paper says freezing hurts a lot). Now end-to-end.
- Old defaults used tuned values: patience 0, warm-up, clip 1.0, batch schedule 64 -> 2048, 100 epochs, RMSE monitor, protein length 1024. Now the code defaults.
- Old defaults skipped the random split groups and used absent groups (`enzyme_structure_splits`, `conformer_cosine_splits`).
- Old results were written inside the split folders. Now they go to `--output_root`.
- The `src.utils.rich_progress` import failed when run from the repo root because `src/` of the baseline shadowed EMULaToR `src/`. The EMULaToR root is now put first on the path.
- Default paths were missing. Now repo-relative via `--task`, with outputs under `outputs/baselines/StructureFree-DTA`.
- `run_split_benchmarks.py` no longer writes `summary_runs.csv` or `summary_by_split.csv`. One process per seed or GPU used to overwrite them. Only `aggregate_results.py` summarises, and it reads every `seed_*` folder.
- `aggregate_results.py` now groups by `split_name` (split folder: group and threshold) by default. `--group_by split_group` pools the folders of a group.
- `commands.txt` names the `mldb` conda env.
- The aggregator grouped runs by a wrong folder level for random and time splits. Layout is now always `<split_group>/<split_name>/seed_N`.

## Deviations
- Final val/test predictions use batch size 256. Eval mode (no dropout, BatchNorm running stats) makes predictions independent of batch size. Validation during training uses 64 as in the original.
- `num_workers` (default 8) for tokenisation in DataLoader workers. Same tokens.
- Train-set predictions are not written (the original does not evaluate train).
- Only the best checkpoint is saved (no per-epoch latest checkpoint).
- No other speed-up. fp16 autocast is the original behaviour.

## Changes after review

- `train_single_target_tvt.py`: each run deletes `logfile.csv` and `bestmodel.pth` at start, so `--overwrite` gives a clean log.
- `--epochs 0` no longer raises `NameError`: `epochs_run` is 0, and the untrained model is evaluated (no best checkpoint to load).
- Smoke (mldb, CPU, 16 rows): `--epochs 0` finished with `epochs_run: 0` and removed a stale log; two `--epochs 1` runs left a 2-line log. Output: `scratchpad/bench_smoke/StructureFree-DTA`.

## Commands
See `emulator_bench/commands.txt`. Full run per GPU:
```
python -m emulator_bench.run_split_benchmarks --task kd --seeds 666 --device cuda:0
python -m emulator_bench.run_split_benchmarks --task kd --seeds 777 --device cuda:1
python -m emulator_bench.run_split_benchmarks --task kd --seeds 888 --device cuda:2
python -m emulator_bench.aggregate_results --task kd
```
Use `--task ec50` etc. for other tasks (the paper covers Kd only).
Env: `/home/adhil/.conda/envs/mldb/bin/python` (torch 2.12, transformers 5.19). Run with cwd = the StructureFree-DTA repo root.
Five Kd split folders, 3 seeds = 15 runs.

## Smoke test
Env mldb, CUDA_VISIBLE_DEVICES=1. `run_split_benchmarks --task kd --split_groups random_splits_grouped_sequence enzyme_sequence_splits --seeds 666 --epochs 2 --max_rows 64`.
Re-run after the fixes: two separate single-seed processes (666, 777) on `enzyme_sequence_splits` and `substrate_splits`, 1 epoch, 64 rows. `aggregate_results` (default grouping) showed `n_seeds = 2` for both folders, so no process overwrote another. Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/StructureFree-DTA/out2`.
Earlier run: both runs finished: logs, checkpoints, predictions, metrics, summary CSVs written to
`/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/StructureFree-DTA`.
`aggregate_results` read them. Nothing was written under EMULaToR/data. Metrics are meaningless (64 rows, 2 epochs).
The two HF encoders were downloaded to `~/.cache/huggingface` during the test.

## Open issues
- Runtime is not measured at full scale. Each run fine-tunes two transformers for up to 150 epochs on about 55k Kd rows.
- The paper appendix describes cached features and bf16 for binding tasks. This wrapper uses fine-tuned encoders and fp16 as in the original code. The camera-ready text may need a change.
- Early stopping uses validation loss (original). The old wrapper used RMSE.
- Only Kd is run by default. Other tasks work through `--task` but the paper does not list them for this model.
- Original `src/`, `cfg/`, `train.sh` are unchanged. `models.py` sets `dropout=0.01` as constructor default, but config and wrapper pass 0.05.
- `inception_out_channels` is accepted by `AffinityPredictor` but unused.
