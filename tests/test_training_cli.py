"""Exercise the actual training, checkpoint, evaluation, export and prediction CLIs."""

import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
        env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result.stdout


def test_train_evaluate_export_predict(tmp_path):
    data = tmp_path / "data"
    for split in ["train", "val"]:
        for label in range(5):
            directory = data / split / str(label)
            directory.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (40, 40), (label*40, i*80, 40)).save(directory / f"{i}.png")
    output = tmp_path / "output"
    common = ["recipes/imagenet_deit/main.py", "--model", "deit_tiny_patch16_LS",
        "--data-path", data, "--output_dir", output, "--device", "cpu",
        "--batch-size", "5", "--epochs", "1", "--warmup-epochs", "0",
        "--num_workers", "0", "--input-size", "64", "--no-repeated-aug", "--no-model-ema",
        "--mixup", "0", "--cutmix", "0", "--lfa", "--lfa-head-type", "independent",
        "--lfa-initial-k", "10", "--lfa-initial-b", "0.5", "--lfa-loss-type", "exponential",
        "--final-loss", "ce", "--no-decay-last-layer"]
    run(common)
    checkpoint = output / "checkpoint.pth"
    assert checkpoint.is_file()
    evaluation = run([*common, "--resume", checkpoint, "--eval"])
    assert "Accuracy of the network" in evaluation
    weights = tmp_path / "model.safetensors"
    run(["tools/export_weights.py", checkpoint, weights, "--trust-checkpoint", "--deit-inference-only"])
    prediction = run(["tools/predict.py", "--weights", weights, "--model", "deit_tiny_patch16_LS",
                      "--input-size", "64", "--image", data / "val/0/0.png"])
    assert len(json.loads(prediction)) == 5
    manifest = json.loads(weights.with_suffix(".json").read_text())
    assert len(manifest["sha256"]) == 64
    assert manifest["deit_auxiliary_heads_removed"]
