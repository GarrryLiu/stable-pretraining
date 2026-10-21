"""Formula, gradient, pooling and API tests for LeVLJEPA (offline, CPU)."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import stable_pretraining as spt
from stable_pretraining.losses import LeVLJEPALoss

pytestmark = pytest.mark.unit


class TinyText(nn.Module):
    """Token encoder exposing the HuggingFace output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 5)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_model(**kwargs) -> spt.Module:
    """Build an injectable encoder configuration without model downloads."""
    return spt.LeVLJEPA(
        vision_encoder=nn.Linear(4, 6),
        text_encoder=TinyText(),
        vision_dim=6,
        text_dim=5,
        embedding_dim=8,
        pre_projection_dims=(),
        predictor_dims=(),
        num_projections=16,
        **kwargs,
    )


def make_batch() -> dict[str, torch.Tensor]:
    """Build nonidentical image/text samples with right and left padding."""
    return {
        "image": torch.randn(4, 4),
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [0, 1, 2, 3], [1, 0, 0, 0], [2, 3, 4, 5]]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]
        ),
    }


def reference_sigreg(x: torch.Tensor, seed: int, projections: int = 16) -> torch.Tensor:
    """Independent literal reference calculation of the sliced EP objective."""
    g = torch.Generator().manual_seed(seed)
    directions = F.normalize(torch.randn(x.shape[1], projections, generator=g), dim=0)
    t = torch.linspace(0, 3, 17)
    phi = (-t.square() / 2).exp()
    phase = (x @ directions).unsqueeze(-1) * t
    error = (phase.cos().mean(0) - phi).square() + phase.sin().mean(0).square()
    # Trapezoidal integral includes the two-sided characteristic-function factor.
    return 2 * torch.trapezoid(error * phi, t, dim=-1).mean() * x.shape[0]


@pytest.mark.parametrize("alignment", ["mse", "cosine"])
def test_reference_formula_and_gradients(alignment):
    torch.manual_seed(8)
    tensors = [torch.randn(7, d, requires_grad=True) for d in (4, 5, 5, 4)]
    image, text, ip, tp = tensors
    criterion = LeVLJEPALoss(
        lambda_vision=0.1,
        lambda_text=0.2,
        alignment=alignment,
        num_projections=16,
        seed=42,
    )
    result = criterion(*tensors)
    if alignment == "mse":
        prediction = (
            F.mse_loss(ip, text.detach()) + F.mse_loss(tp, image.detach())
        ) / 2
    else:
        prediction = (
            (F.normalize(ip, dim=-1) - F.normalize(text.detach(), dim=-1))
            .square()
            .sum(-1)
            .mean()
            + (F.normalize(tp, dim=-1) - F.normalize(image.detach(), dim=-1))
            .square()
            .sum(-1)
            .mean()
        ) / 2
    expected = (
        0.7 * prediction
        + 0.1 * reference_sigreg(image, 42)
        + 0.2 * reference_sigreg(text, 43)
    )
    torch.testing.assert_close(result["loss"], expected)
    actual_grad = torch.autograd.grad(result["loss"], tensors, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tensors)
    for actual, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected_value)


def test_stop_gradient_targets_but_regularizer_trains_encoders():
    tensors = [torch.randn(4, 8, requires_grad=True) for _ in range(4)]
    result = LeVLJEPALoss(num_projections=16)(*tensors)
    result["prediction_loss"].backward(retain_graph=True)
    assert tensors[0].grad is None and tensors[1].grad is None
    assert tensors[2].grad.abs().sum() > 0 and tensors[3].grad.abs().sum() > 0
    (result["sigreg_vision"] + result["sigreg_text"]).backward()
    assert tensors[0].grad.abs().sum() > 0 and tensors[1].grad.abs().sum() > 0


def test_dict_api_pooling_and_all_modules_receive_gradients():
    model, batch = make_model(), make_batch()
    result = model(batch, stage="fit")
    assert "loss" not in batch
    assert result["embedding"].shape == (4, 6)
    assert result["image_prediction"].shape == (4, 8)
    expected_raw = model.text_encoder.embedding(torch.tensor([3, 3, 1, 5]))
    torch.testing.assert_close(
        result["text_embedding"], model.text_pre_proj(expected_raw)
    )
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_checkpoint_reproduces_next_random_slices_and_eval_does_not_advance():
    model, batch = make_model(), make_batch()
    model(batch, stage="fit")
    saved = copy.deepcopy(model.state_dict())
    expected = model(batch, stage="fit")["loss"]
    restored = make_model()
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored(batch, stage="fit")["loss"], expected)
    restored.eval()
    before = restored.levljepa_loss.vision_regularizer.step.clone()
    torch.testing.assert_close(
        restored(batch, stage="validate")["loss"], restored(batch, stage="test")["loss"]
    )
    assert torch.equal(before, restored.levljepa_loss.vision_regularizer.step)
    assert "loss" not in restored(batch, stage="predict")
    assert restored({"image": batch["image"]}, stage="predict")["embedding"].shape == (
        4,
        6,
    )


@pytest.mark.parametrize("value", [0.0, 100.0])
def test_collapsed_bfloat16_embeddings_remain_finite(value):
    tensors = [
        torch.full((8, 16), value, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(4)
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = LeVLJEPALoss(num_projections=16)(*tensors)
    assert result["loss"].dtype == torch.float32
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(t.grad).all() for t in tensors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_vision": -0.1},
        {"lambda_text": float("nan")},
        {"lambda_vision": 0.8, "lambda_text": 0.2},
        {"alignment": "other"},
        {"num_projections": 0},
        {"num_knots": 2},
    ],
)
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError):
        LeVLJEPALoss(**kwargs)


def test_bad_stage_shapes_and_empty_text_fail_clearly():
    model, batch = make_model(), make_batch()
    with pytest.raises(ValueError, match="stage"):
        model(batch, stage="typo")
    with pytest.raises(ValueError, match="paired"):
        model({"image": batch["image"]}, stage="fit")
    batch["attention_mask"][0] = 0
    with pytest.raises(ValueError, match="unmasked"):
        model(batch, stage="fit")
    with pytest.raises(ValueError, match="shapes"):
        LeVLJEPALoss(num_projections=16)(
            torch.randn(4, 3), torch.randn(4, 5), torch.randn(4, 3), torch.randn(4, 3)
        )


def test_composable_forward_matches_method():
    model, batch = make_model().eval(), make_batch()
    components = {
        name: copy.deepcopy(getattr(model, name))
        for name in (
            "vision_encoder",
            "text_encoder",
            "vision_pre_proj",
            "text_pre_proj",
            "projector_vision",
            "projector_text",
            "levljepa_loss",
        )
    }
    composed = spt.Module(forward=spt.forward.levljepa, **components).eval()
    torch.testing.assert_close(
        model(batch, stage="validate")["loss"],
        composed(batch, stage="validate")["loss"],
    )


def test_normal_distribution_has_lower_sigreg_than_collapse():
    torch.manual_seed(73)
    normal = torch.randn(512, 32)
    collapsed = torch.zeros_like(normal)
    criterion = LeVLJEPALoss(num_projections=64).eval()
    normal_stat = criterion.vision_regularizer(normal)
    collapsed_stat = criterion.vision_regularizer(collapsed)
    assert normal_stat < collapsed_stat / 10


def test_batchnorm_heads_under_bfloat16_autocast():
    model = spt.LeVLJEPA(
        vision_encoder=nn.Linear(4, 6),
        text_encoder=TinyText(),
        vision_dim=6,
        text_dim=5,
        embedding_dim=8,
        pre_projection_dims=(16,),
        predictor_dims=(16, 16),
        num_projections=16,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(make_batch(), stage="fit")
    assert output["loss"].dtype == torch.float32
    output["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_example_evaluation_restores_training_mode():
    from examples.levljepa_flickr8k import evaluate

    class Rows(list):
        captions = ["one", "two", "three", "four"]

    batch = make_batch()
    rows = Rows([{key: value[i] for key, value in batch.items()} for i in range(4)])
    loader = torch.utils.data.DataLoader(rows, batch_size=4)
    model = make_model().train()
    evaluate(model, loader, torch.device("cpu"))
    assert model.training and model.levljepa_loss.vision_regularizer.training
    model(batch, stage="fit")
    assert int(model.levljepa_loss.vision_regularizer.step) == 1
    model.eval()
    evaluate(model, loader, torch.device("cpu"))
    assert not model.training
