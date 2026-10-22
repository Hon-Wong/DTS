import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


small = load("small_recipe", "recipes/small_scale/main.py")
losses = load("deit_losses", "recipes/imagenet_deit/losses.py")
models = load("deit_models", "recipes/imagenet_deit/models_v2.py")
export = load("export_weights", "tools/export_weights.py")


@pytest.mark.parametrize("beta", [0.0, 0.5, 3.0])
def test_targets_match_logit_margin(beta):
    y = torch.tensor([0, 3])
    criterion = small.TrajectoryCriterion("exponential", 5, g_max=10, beta=beta)
    for t in [0, 0.25, 0.5, 1]:
        target = criterion.get_target(t, y)
        alpha = t if beta == 0 else math.expm1(beta*t)/math.expm1(beta)
        expected = F.softmax(10 * alpha * F.one_hot(y, 5).float(), -1)
        torch.testing.assert_close(target, expected)
        torch.testing.assert_close(target.sum(-1), torch.ones(2))
        assert losses.get_interpolation_alpha(t, "exponential", beta) == pytest.approx(alpha)


@pytest.mark.parametrize("final_loss", ["ce", "bce"])
def test_deit_mixup_loss_matches_explicit_equation(final_loss):
    torch.manual_seed(1)
    logits = torch.randn(4, 2, 5, requires_grad=True)
    targets = torch.tensor([[0.7, 0.3, 0, 0, 0], [0, 0, 0.4, 0, 0.6]])
    criterion = losses.AdaptiveLogitFlowLoss(5, 4, fixed_scale=10, fixed_bias=0.5,
        init_lfa_scale=0.5, decay_lfa_scale=True, decay_last_layer=False, final_loss=final_loss)
    criterion.set_progress(0.4)
    actual = criterion(logits, targets)
    auxiliary = []
    for i in range(4):
        gap = 10 * math.expm1(0.5*(i+1)/4) / math.expm1(0.5)
        target_logits = (gap + (targets + 1e-8).log()) * (targets > 0)
        auxiliary.append(-(target_logits.softmax(-1) * logits[i].log_softmax(-1)).sum(-1).mean())
    terminal = F.cross_entropy(logits[-1], targets) if final_loss == "ce" else F.binary_cross_entropy_with_logits(logits[-1], (targets > 0).float())
    expected = terminal + 0.3 * torch.stack(auxiliary).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(dim=(1, 2)) > 0).all()


def test_fisher_rao_branch_is_finite():
    logits = torch.randn(4, 2, 5, requires_grad=True)
    criterion = losses.AdaptiveLogitFlowLoss(5, 4, fixed_scale=10, fixed_bias=0.5,
                                           schedule_type="fisher_rao")
    criterion(logits, torch.tensor([0, 3])).backward()
    assert torch.isfinite(logits.grad).all()


def test_deit_export_preserves_terminal_prediction(tmp_path):
    from safetensors.torch import save_file, load_file
    kwargs = dict(img_size=32, patch_size=8, num_classes=5, embed_dim=24,
                  depth=3, num_heads=3, mlp_ratio=2, drop_rate=0.1)
    supervised = models.vit_models(**kwargs, lfa=True, head_type="independent").eval()
    inference = models.vit_models(**kwargs, lfa=False).eval()
    exported = export.extract_state({"model": supervised.state_dict()}, deit_inference_only=True)
    path = tmp_path / "model.safetensors"
    save_file(exported, str(path))
    inference.load_state_dict(load_file(str(path)), strict=True)
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(supervised(images)[-1], inference(images))
    assert not any(k.startswith("aux_") for k in exported)


def test_small_scale_backward_and_checkpoint_roundtrip():
    model = small.TrajectoryTransformer("exponential", num_layers=3, embed_dim=24,
        patch_size=4, img_size=16, num_classes=5, num_heads=3, mlp_dim=48)
    logits, _ = model(torch.randn(2, 3, 16, 16))
    loss = small.TrajectoryCriterion("exponential", 5, g_max=10, beta=0.5)(
        logits, torch.tensor([1, 3]), lambd=0.5)
    loss.backward()
    assert torch.isfinite(model.patch_embed.weight.grad).all()


@pytest.mark.parametrize("entry", ["recipes/small_scale/main.py", "recipes/imagenet_deit/main.py",
    "recipes/imagenet_convnext/main.py", "recipes/probing/probe_pretrained.py",
    "tools/train.py", "tools/predict.py", "tools/export_weights.py"])
def test_entrypoint_help(entry):
    result = subprocess.run([sys.executable, str(ROOT / entry), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("config", sorted((ROOT / "configs").glob("*.json")))
def test_portable_recipe(config):
    result = subprocess.run([sys.executable, str(ROOT / "tools/train.py"),
        "--recipe", config.stem, "--data-path", "/tmp/data with spaces", "--output-dir", "/tmp/dts-output",
        "--finetune", "/tmp/model.pth", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "{data_path}" not in result.stdout and "{finetune}" not in result.stdout
    assert "/p/scratch/" not in config.read_text()


def test_export_rejects_training_payload_without_weights():
    with pytest.raises(ValueError):
        export.extract_state({"epoch": 2, "optimizer": {}})
