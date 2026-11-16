"""Checks that the emulator_bench wrapper reproduces the original training path.

Run `emulator_bench/verify_equivalence.sh` (CPU) or add `--gpu`. No downloads: the tokenizers are the public
ESM2 and ChemBERTa tokenizer files in `tests/fixtures`. The encoders are tiny random models of the same
classes (EsmModel, RobertaModel), saved to a temporary directory and loaded by `AutoModel.from_pretrained`.
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmConfig, EsmModel, RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emulator_bench import train_single_target_tvt as wrapper  # noqa: E402  (also puts src/ on sys.path)
from dataloader import DrugTargetDataset, collate_fn  # noqa: E402  (original code)
from models import AffinityPredictor, DrugTargetInteractionLoss  # noqa: E402  (original code)

FIXTURES = Path(__file__).parent / "fixtures"
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
    return (AutoTokenizer.from_pretrained(FIXTURES / "molecule_tokenizer"), AutoTokenizer.from_pretrained(FIXTURES / "protein_tokenizer"))


@pytest.fixture(scope="module")
def encoder_dirs(tmp_path_factory):
    """Tiny random encoders. The vocabularies match the tokenizers; the real encoders are larger."""
    root = tmp_path_factory.mktemp("encoders")
    torch.manual_seed(0)
    EsmModel(EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       pad_token_id=1, mask_token_id=32, max_position_embeddings=130)).save_pretrained(root / "protein")
    RobertaModel(RobertaConfig(vocab_size=600, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                               pad_token_id=1, max_position_embeddings=130)).save_pretrained(root / "molecule")
    return str(root / "protein"), str(root / "molecule")


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


def test_gradient_checkpointing_off_matches_on(tensors, encoder_dirs):
    """One forward and backward in train mode: same loss and gradients with checkpointing on and off."""
    torch.manual_seed(0)  # the head init must not depend on the tests that ran before
    model = AffinityPredictor(*encoder_dirs, [1024, 768, 512, 256, 1], 256, 0.05)
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


@pytest.mark.gpu
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
        if fused and os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1":  # negative control: the test must fail
            with torch.no_grad():
                for p in net.parameters():
                    p.mul_(1.05)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-5, fused=fused)
        for _ in range(60):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(x).view(-1), y).backward()
            opt.step()
        finals.append(torch.cat([p.detach().flatten() for p in net.parameters()]))
    torch.testing.assert_close(finals[1], finals[0], rtol=1e-3, atol=1e-5)


class RecordingLoss:
    def __init__(self, loss_fn):
        self.loss_fn, self.values = loss_fn, []

    def __call__(self, output, labels):
        loss = self.loss_fn(output, labels)
        self.values.append(loss.item())
        return loss


K = 10  # the production gap may be 10 times the fp32 noise (the production run with initial weights perturbed by one fp16 rounding)
U = 2**-11  # one rounding of the weights at the production precision (fp16)


def perturb(module, rel=U):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))


def train_steps(encoder_dirs, data, production, noise=False, scale=1.0):
    """3 epochs of 8 batches. Production: fp16 autocast, GradScaler, fused AdamW, no checkpointing.
    Reference: fp32, plain AdamW, original gradient checkpointing."""
    device = torch.device("cuda")
    torch.manual_seed(1)
    model = AffinityPredictor(*encoder_dirs, [1024, 768, 512, 256, 1], 256, 0.0).to(device)  # no dropout: masks differ between the paths
    if noise:
        perturb(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    if production:
        model.protein_model.gradient_checkpointing_disable()
        model.molecule_model.gradient_checkpointing_disable()
    loss_fn = RecordingLoss(DrugTargetInteractionLoss(alpha=0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5, fused=production)
    # Start at the scale the default one settles at (2**16 halves 7 times in these 24 steps and skips them): the toy run then models steady state.
    scaler = torch.amp.GradScaler("cuda", init_scale=2**9, enabled=production)
    amp_dtype = torch.float16 if production else None
    loader = wrapper.GpuBatches(data, 4, shuffle=True)
    torch.manual_seed(2)
    for _ in range(3):
        wrapper.train_one_epoch(model, loader, loss_fn, optimizer, scaler, amp_dtype, device, "")
    skipped = production and scaler.get_scale() != 2**9  # a skipped step halves the scale
    _, _, val_loss = wrapper.evaluate(model, wrapper.GpuBatches(data, 8, shuffle=False), loss_fn.loss_fn, device)
    return torch.tensor(loss_fn.values, dtype=torch.float64), val_loss, skipped  # the evaluation uses the unwrapped loss


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 and fused AdamW need CUDA")
def test_production_precision_stays_close_to_fp32(collate, encoder_dirs):
    """Production training (fp16 autocast, GradScaler, fused AdamW) against fp32 with plain AdamW, on the same
    seed and batches.

    The reference is fp32; the noise control is the production run with initial weights perturbed by 2**-11 relative.
    Bounds: max per-step loss gap and the final validation loss gap each within K times their noise floor;
    the step-0 loss gap within K times its floor. The scaler must skip no step.
    With EQUIV_NEGATIVE_CONTROL=1 the production weights are scaled by 1.05 for the compared run; the test must fail.
    """
    n = 32
    frame = pd.DataFrame({"smiles": [SMILES[i % len(SMILES)] for i in range(n)], "sequence": [PROTEINS[i % len(PROTEINS)] for i in range(n)],
                          "y": [float(i % 7) - 3.0 for i in range(n)]})
    data = collate.tokenize(frame, Namespace(smiles_col="smiles", sequence_col="sequence", target_col="y"), torch.device("cuda"))
    (ref, ref_val, _), (noise, noise_val, noise_skipped), (prod, prod_val, prod_skipped) = (
        train_steps(encoder_dirs, data, production=False),
        train_steps(encoder_dirs, data, production=True, noise=True),
        train_steps(encoder_dirs, data, production=True),
    )
    if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1":
        cand, cand_val, cand_skipped = train_steps(encoder_dirs, data, production=True, scale=1.05)
    else:
        cand, cand_val, cand_skipped = prod, prod_val, prod_skipped
    assert len(ref) == len(noise) == len(prod) == len(cand) == 24
    floor, gap = (noise - prod).abs().max().item(), (cand - ref).abs().max().item()
    tiny, tiny0 = 1e-7 * ref.abs().max().item(), 1e-7 * ref[0].abs().item()
    gap0, floor0 = (cand[0] - ref[0]).abs().item(), (noise[0] - prod[0]).abs().item()
    final_gap, final_floor = abs(cand_val - ref_val), abs(noise_val - prod_val)
    print(
        f"sfdta fp16: max gap={gap:.3e}, max floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}, "
        f"step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}; final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * abs(ref_val))
    assert not (noise_skipped or prod_skipped or cand_skipped), "the scaler skipped a step"
