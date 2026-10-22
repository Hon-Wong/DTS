"""Launch a checked-in recipe with machine-independent paths."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(args, extra):
    config_path = ROOT / "configs" / (args.recipe + ".json")
    config = json.loads(config_path.read_text())
    replacements = {"{data_path}": str(Path(args.data_path).resolve()),
                    "{output_dir}": str(Path(args.output_dir).resolve())}
    if "{finetune}" in config["arguments"]:
        if not args.finetune:
            raise ValueError("This recipe requires --finetune PATH to pretrained weights.")
        replacements["{finetune}"] = str(Path(args.finetune).resolve())
    argv = [replacements.get(value, value) for value in config["arguments"]]
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(args.nproc_per_node),
               str(ROOT / config["entrypoint"]), *argv, *extra]
    return command, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True,
                        choices=sorted(p.stem for p in (ROOT / "configs").glob("*.json")))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finetune")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    command, config = build_command(args, extra)
    print(shlex.join(command), flush=True)
    if config.get("reference_world_size", args.nproc_per_node) != args.nproc_per_node:
        print("Note: the archived recipe used %s GPUs; per-GPU batch size is unchanged."
              % config["reference_world_size"], flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
