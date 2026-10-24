"""Checks that the emulator_bench wrapper reproduces the original training path (CPU, fp32).

Run from the StructureFree-DTA root:
    CUDA_VISIBLE_DEVICES= python -m pytest tests/test_original_equivalence.py -v
The encoders must be in the local Hugging Face cache. Otherwise the tests skip.
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emulator_bench import train_single_target_tvt as wrapper  # noqa: E402  (also puts src/ on sys.path)
from dataloader import DrugTargetDataset, collate_fn  # noqa: E402  (original code)
from models import AffinityPredictor, DrugTargetInteractionLoss  # noqa: E402  (original code)

PROTEIN_MODEL = "facebook/esm2_t6_8M_UR50D"
MOLECULE_MODEL = "DeepChem/ChemBERTa-77M-MLM"
MAX_LEN = 128
BATCH = 4

# Short, long (truncated), repeated and mixed-length strings.
SMILES = ["CCO", "c1ccccc1O", "CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "C" * 300, "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
          "C" * 300, "N", "c1ccccc1O", "CCN(CC)CC"]
PROTEINS = ["MKV", "M" + "ACDEFGHIKLMNPQRSTVWY" * 10, "MKVLAAGIVGLLLAQ", "MKV", "MKVLAAGIVGLLLAQ",
            "M" + "ACDEFGHIKLMNPQRSTVWY" * 10, "A" * (MAX_LEN - 2), "A" * (MAX_LEN - 1), "A" * MAX_LEN, "MKV"]
LABELS = [float(i) + 0.25 for i in range(len(SMILES))]


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import AutoTokenizer

    try:
        return (AutoTokenizer.from_pretrained(MOLECULE_MODEL, local_files_only=True),
                AutoTokenizer.from_pretrained(PROTEIN_MODEL, local_files_only=True))
    except OSError:
        pytest.skip("Tokenizers are not in the local Hugging Face cache.")


@pytest.fixture(scope="module")
def collate(tokenizers):
    obj = wrapper.Collate.__new__(wrapper.Collate)
    obj.molecule_tokenizer, obj.protein_tokenizer = tokenizers
    obj.max_molecule_length = obj.max_protein_length = MAX_LEN
    return obj


@pytest.fixture(scope="module")
def tensors(collate):
    frame = pd.DataFrame({"smiles": SMILES, "sequence": PROTEINS, "y": LABELS})
    args = Namespace(smiles_col="smiles", sequence_col="sequence", target_col="y")
    return collate.tokenize(frame, args, torch.device("cpu"))


def original_batch(tokenizers, indices):
    dataset = DrugTargetDataset(SMILES, PROTEINS, LABELS)
    return collate_fn([dataset[i] for i in indices], *tokenizers, MAX_LEN, MAX_LEN)


def test_token_table_equals_original_collate(tokenizers, tensors):
    """Every row, and several index orders, give identical ids, masks and labels."""
    n = len(SMILES)
    for indices in (list(range(n)), list(reversed(range(n))), [4, 0, 4, 3], [1, 5], [8, 9, 6, 7], [0]):
        expected = original_batch(tokenizers, indices)
        index = torch.tensor(indices)
        for key, value in expected.items():
            assert value.dtype == tensors[key].dtype, key
            assert torch.equal(tensors[key][index], value), (key, indices)


def test_loader_order_equals_original_dataloader(tokenizers, tensors):
    """Same seed gives the same shuffle, so each epoch yields the same batches as the original loader."""
    original = DataLoader(DrugTargetDataset(SMILES, PROTEINS, LABELS), batch_size=BATCH, shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch, *tokenizers, MAX_LEN, MAX_LEN))
    ours = wrapper.GpuBatches(tensors, BATCH, shuffle=True)
    assert len(ours) == len(original)
    torch.manual_seed(666)
    expected = [list(original) for _ in range(3)]
    torch.manual_seed(666)
    actual = [list(ours) for _ in range(3)]
    for epoch_expected, epoch_actual in zip(expected, actual):
        for want, got in zip(epoch_expected, epoch_actual):
            assert want.keys() == got.keys()
            for key in want:
                assert torch.equal(want[key], got[key]), key


def test_gradient_checkpointing_off_matches_on(tensors):
    """One forward and backward in train mode: same loss and gradients with checkpointing on and off."""
    try:
        model = AffinityPredictor(PROTEIN_MODEL, MOLECULE_MODEL, [1024, 768, 512, 256, 1], 256, 0.05)
    except OSError:
        pytest.skip("Encoders are not in the local Hugging Face cache.")
    loss_fn = DrugTargetInteractionLoss(alpha=0.5)
    batch = {key: value[:BATCH] for key, value in tensors.items()}
    labels = batch.pop("labels")

    def run():
        model.train()
        model.zero_grad()
        torch.manual_seed(0)
        loss = loss_fn(model(batch).view(-1), labels)
        loss.backward()
        return loss.detach().clone(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    assert model.protein_model.is_gradient_checkpointing and model.molecule_model.is_gradient_checkpointing
    loss_on, grads_on = run()
    model.protein_model.gradient_checkpointing_disable()
    model.molecule_model.gradient_checkpointing_disable()
    assert not model.protein_model.is_gradient_checkpointing and not model.molecule_model.is_gradient_checkpointing
    loss_off, grads_off = run()

    torch.testing.assert_close(loss_off, loss_on, rtol=1e-5, atol=1e-6)
    assert grads_on.keys() == grads_off.keys()
    for name in grads_on:
        torch.testing.assert_close(grads_off[name], grads_on[name], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
def test_fused_adamw_close_to_plain_adamw():
    """Known difference: the wrapper uses fused AdamW on CUDA. It is not bit-identical to the default
    kernel (REPORT: the loss differs in the third decimal after 60 steps). Check only that the
    parameters stay close after 60 steps."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Mish(), torch.nn.Linear(64, 1)).cuda()
    x, y = torch.randn(64, 32, device="cuda"), torch.randn(64, device="cuda")
    copies = [{k: v.clone() for k, v in net.state_dict().items()} for _ in range(2)]
    finals = []
    for fused, state in zip((False, True), copies):
        net.load_state_dict(state)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-5, fused=fused)
        for _ in range(60):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(x).view(-1), y).backward()
            opt.step()
        finals.append(torch.cat([p.detach().flatten() for p in net.parameters()]))
    torch.testing.assert_close(finals[1], finals[0], rtol=1e-3, atol=1e-5)
