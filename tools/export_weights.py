"""Export a trusted research checkpoint to tensor-only safetensors weights."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def extract_state(payload, key="model", deit_inference_only=False):
    state = payload[key] if isinstance(payload, dict) and key in payload else payload
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Expected a state dict or a checkpoint containing the selected state-dict key.")
    clean = {}
    for key, tensor in state.items():
        name = key.removeprefix("module.")
        if deit_inference_only and name.startswith(("aux_heads.", "aux_norms.")):
            continue
        if name in clean:
            raise ValueError("Duplicate parameter after prefix removal: " + name)
        clean[name] = tensor.detach().cpu().contiguous().clone()
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key", default="model")
    parser.add_argument("--deit-inference-only", action="store_true")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow pickle objects in a checkpoint you trust.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new filename.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=not args.trust_checkpoint)
    state = extract_state(payload, args.key, args.deit_inference_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output), metadata={"format": "pt"})
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {"filename": args.output.name, "sha256": digest,
               "bytes": args.output.stat().st_size, "tensor_count": len(state),
               "parameters_and_buffers": sum(t.numel() for t in state.values()),
               "deit_auxiliary_heads_removed": args.deit_inference_only}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
