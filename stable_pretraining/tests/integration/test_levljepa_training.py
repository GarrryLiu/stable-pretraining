"""Real Manager training/resume and two-process DDP gradient checks."""

import copy
import os
import tempfile
from pathlib import Path

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.tests.unit.test_levljepa import make_batch, make_model


def _make_real_encoder_model() -> spt.Module:
    from timm.models.vision_transformer import VisionTransformer
    from transformers import GPT2Config, GPT2Model

    return spt.LeVLJEPA(
        vision_encoder=VisionTransformer(
            img_size=16,
            patch_size=8,
            embed_dim=16,
            depth=1,
            num_heads=2,
            num_classes=0,
        ),
        text_encoder=GPT2Model(
            GPT2Config(
                vocab_size=16,
                n_positions=4,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_inner=32,
                attn_pdrop=0,
                resid_pdrop=0,
                embd_pdrop=0,
            )
        ),
        vision_dim=16,
        text_dim=16,
        embedding_dim=8,
        pre_projection_dims=(16,),
        predictor_dims=(16, 16),
        num_projections=16,
    )


class _Pairs(Dataset):
    def __init__(self, real_encoders: bool = False) -> None:
        self.batch = make_batch()
        if real_encoders:
            self.batch["image"] = torch.randn(4, 3, 16, 16)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}


@pytest.mark.integration
@pytest.mark.parametrize("real_encoders", [False, True])
def test_manager_training_and_full_checkpoint_resume(tmp_path, real_encoders):
    spt.set(cache_dir=str(tmp_path / "cache"))
    torch.set_num_threads(1)
    pl.seed_everything(123)
    loader = DataLoader(_Pairs(real_encoders), batch_size=4)
    data = spt.data.DataModule(train=loader, val=loader)
    factory = _make_real_encoder_model if real_encoders else make_model
    model = factory()
    initial = copy.deepcopy(model.vision_encoder.state_dict())
    trainer = pl.Trainer(
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=123)()
    assert trainer.global_step == 3
    assert any(
        not torch.equal(initial[k], v)
        for k, v in model.vision_encoder.state_dict().items()
    )
    checkpoint = str((tmp_path / "resume.ckpt").resolve())
    trainer.save_checkpoint(checkpoint)
    before = int(model.levljepa_loss.vision_regularizer.step)
    restored = factory()
    resumed = pl.Trainer(
        max_steps=5,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    spt.Manager(
        trainer=resumed,
        module=restored,
        data=data,
        seed=123,
        ckpt_path=checkpoint,
        weights_only=False,
    )()
    assert resumed.global_step == 5
    assert int(restored.levljepa_loss.vision_regularizer.step) == before + 2
    assert all(torch.isfinite(p).all() for p in restored.parameters())
    if real_encoders:
        counts = [
            int(layer.num_batches_tracked)
            for layer in restored.modules()
            if isinstance(layer, torch.nn.BatchNorm1d)
        ]
        assert counts and all(count == 5 for count in counts)
    state = resumed.optimizers[0].state_dict()["state"]
    assert state and all(int(value["step"]) == 5 for value in state.values())


def _ddp_worker(rank: int, init_file: str, real_encoders: bool = False) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(19)
    if real_encoders:
        model = _make_real_encoder_model()
        reference = _make_real_encoder_model()
        reference.levljepa_loss.vision_regularizer.distributed = False
        reference.levljepa_loss.text_regularizer.distributed = False
    else:
        model = make_model(distributed=True)
        reference = make_model(distributed=False)
    reference.load_state_dict(model.state_dict())
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(29)
        first, second = make_batch(), make_batch()
        if real_encoders:
            first["image"] = torch.randn(4, 3, 16, 16)
            second["image"] = torch.randn(4, 3, 16, 16)
        full = {key: torch.cat([first[key], second[key]]) for key in first}
        local = first if rank == 0 else second
        # Deliberately different RNG histories must not change the projection slices.
        torch.manual_seed(100 + rank)
        torch.randn(rank * 31)
        wrapped = DistributedDataParallel(model)
        output = wrapped(local, stage="fit")
        if real_encoders:
            # Ordinary DDP BatchNorm uses rank-local statistics, not global ones.
            parts = [reference(batch, stage="predict") for batch in (first, second)]
            expected = reference.levljepa_loss(
                *[
                    torch.cat([part[key] for part in parts])
                    for key in (
                        "image_embedding",
                        "text_embedding",
                        "image_prediction",
                        "text_prediction",
                    )
                ]
            )
        else:
            expected = reference(full, stage="fit")
        baseline = (
            0.96 * expected["prediction_loss"]
            + 0.02 * (expected["sigreg_vision"] + expected["sigreg_text"]) / 2
        )
        mean_loss = output["loss"].detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(
            mean_loss / 2, baseline.detach(), rtol=2e-5, atol=2e-6
        )
        output["loss"].backward()
        baseline.backward()
        for (name, actual), (_, target) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            assert actual.grad is not None, name
            torch.testing.assert_close(
                actual.grad, target.grad, rtol=2e-4, atol=2e-6, msg=name
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.distributed
@pytest.mark.parametrize("real_encoders", [False, True])
def test_real_two_process_ddp_matches_concatenated_batch_gradients(real_encoders):
    if os.name == "nt" or not dist.is_gloo_available():
        pytest.skip("CPU gloo required")
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            _ddp_worker,
            args=(str(Path(directory) / "init"), real_encoders),
            nprocs=2,
            join=False,
        )
        try:
            if not context.join(timeout=45):
                if not context.join(timeout=5):
                    raise AssertionError("LeVLJEPA DDP test timed out")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


@pytest.mark.integration
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS required")
def test_real_encoders_batchnorm_heads_on_mps():
    torch.manual_seed(123)
    model = _make_real_encoder_model().to("mps")
    batch = make_batch()
    batch["image"] = torch.randn(4, 3, 16, 16)
    batch = {key: value.to("mps") for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    before = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(batch, stage="fit")["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in model.named_parameters()
    )
    assert int(model.levljepa_loss.vision_regularizer.step) == 2
