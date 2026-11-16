# emulator_bench

Multi-seed evaluation of StructureFree-DTA with its original published settings on the default EMULaToR splits.

The model is the original `src/` code: ESM2-t6-8M and ChemBERTa-77M-MLM encoders (both fine-tuned), two Residual Inception blocks, and a 4-layer Mish MLP. Training follows `cfg/config.yaml` and `src/main.py`. No hyperparameter is tuned.

## Data

The wrapper is Kd-only, as in the paper. It reads `data/processed/datasets/binding_affinity_dataset/kd`. Use `--base_dir` for another split root.

Split groups run by default: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`, `substrate_splits/threshold_*`, `conformer_cosine_splits/threshold_*`. Missing groups are skipped. Each folder holds `train/val/test.parquet` with `sequence`, `smiles`, `log10_value`.

## Scripts

- `train_single_target_tvt.py`: trains one split and seed. Writes `bestmodel.pth`, `logfile.csv`, `pred_label_{val,test}.csv`, `final_results_{val,test}.csv`, `run_summary.json`.
- `run_split_benchmarks.py`: runs all discovered splits and seeds (default 666 777 888). Skips finished runs. `--gpus 0 1 2` gives each run one GPU through `CUDA_VISIBLE_DEVICES`. `--runs_per_gpu N` sets the runs per GPU. Without `--gpus`, runs go one at a time on `--device`. `--amp` switches the training precision from the original fp16 autocast to bf16 (Ampere or newer; fp16 with GradScaler on older GPUs). Run folders are separate, so parallel runs never overwrite each other.
- `aggregate_results.py`: prints mean and std over seeds from an output root.

Outputs go to `--output_root`, default `outputs/baselines/StructureFree-DTA/kd`, as `<split_group>/<split_name>/seed_<seed>`. Split groups without `threshold_*` folders use `all` as `<split_name>`. Split files are never modified.

## Settings

See `REPORT.md` for the settings table and the paper/code differences. `--epochs` and `--max_rows` are for smoke tests only.

See `commands.txt` for run commands.

## Reproduce

`run.sh` runs the whole chain for the paper results: training and the seed summary. This wrapper has no cache step. It covers Kd on all default splits with seeds 666, 777 and 888. It uses the default fp16 autocast (no `--amp`), as the paper runs did. Set `PY` (we used the `mldb` conda env), `GPUS`, `SEEDS`, `RUNS_PER_GPU`, `DATA_ROOT` and `OUT_ROOT` to change the setup. Run it from any folder. `--dry-run` prints the commands. Train also evaluates and writes the test metrics.

```
emulator_bench/run.sh [--dry-run] [train|aggregate|all]
```

## Verify equivalence

```
emulator_bench/verify_equivalence.sh          # CPU, fp32, about 10 s
emulator_bench/verify_equivalence.sh --gpu    # also the fp16 and fused AdamW tests; needs one CUDA device
```

Expected env: `mldb` (torch, transformers, pandas, pytest). The script does not create it. It fails if any CPU test is skipped. It needs no private data and no downloads: the tokenizers are the public ESM2 and ChemBERTa tokenizer files in `tests/fixtures`, and the encoders are tiny random `EsmModel` and `RobertaModel` networks.

What `tests/test_original_equivalence.py` compares:
- Tokens: the wrapper's token table against the original `collate_fn`, for several index orders. Ids, masks and labels must be identical.
- Loader: `GpuBatches` against the original `DataLoader` with the same seed, for three epochs. Batches must be identical.
- Gradient checkpointing: one training step with checkpointing on (original) and off (wrapper). Loss within rtol 1e-5, gradients within rtol 1e-4.
- `--gpu`, fused AdamW: 60 steps against plain AdamW. Parameters within rtol 1e-3.
- `--gpu`, production precision: 3 epochs with fp16 autocast, GradScaler, fused AdamW and no checkpointing against fp32 with plain AdamW. The noise control is the production run with initial weights perturbed by one rounding of the production dtype (2**-11, fp16). Per step, the gap |production - fp32| must be at most 10 times the floor |perturbed production - production|. The same holds for the final validation loss. Step 0 is judged like every other step: its gap must be at most 10 times its floor. The toy run sets dropout to 0 (the masks differ between the paths) and starts GradScaler at 2**9, the scale it settles at (the default 2**16 skips 7 of the 24 steps). The scaler must skip no step. Each test prints `max gap`, `max floor` and `ratio`; run with `-s` to see the line.

Negative control: `emulator_bench/verify_equivalence.sh --negative-control` scales the weights (also in the fused AdamW test) of the compared (production) run by 1.05 and runs only the GPU tests. It exits 0 only if every GPU test fails, so a passing GPU test shows a bound too loose to catch a 5% error.

Not exact: fused AdamW and fp16 training (the production setting) differ from the original by rounding. The tests use tiny random encoders, so they do not check the pretrained weights.
