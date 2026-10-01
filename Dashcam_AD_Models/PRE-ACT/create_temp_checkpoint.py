import argparse
import os
import shlex
import sys

import torch

import train_autoencoder_anticipation_ddp as train_module
from models import build_model


class _Captured(Exception):
    pass


def read_train_argv(script_path):
    with open(script_path, "r") as f:
        lines = f.read().splitlines()

    variables = {}
    command = None
    for line in lines:
        line = line.strip()
        if line.startswith("model_id="):
            variables["model_id"] = line.split("=", 1)[1].strip().strip('"')
        if line.startswith("torchrun"):
            command = line

    for key, value in variables.items():
        command = command.replace("${" + key + "}", value).replace("$" + key, value)

    tokens = shlex.split(command)
    idx = next(i for i, t in enumerate(tokens) if t.endswith("train_autoencoder_anticipation_ddp.py"))
    return tokens[idx + 1:]


def capture_train_args(argv):
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def patched(self, args=None, namespace=None):
        captured["args"] = original(self, args, namespace)
        raise _Captured

    argparse.ArgumentParser.parse_args = patched
    sys.argv = ["train_autoencoder_anticipation_ddp.py"] + argv
    try:
        train_module.main()
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["args"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", type=str, default="./scripts/train_dad.sh")
    parser.add_argument("--name", type=str, default="best_mauc_0_1.pt")
    cli = parser.parse_args()

    train_args = capture_train_args(read_train_argv(cli.train_script))
    model = build_model(train_args)

    os.makedirs(train_args.output_dir, exist_ok=True)
    out_path = os.path.join(train_args.output_dir, cli.name)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(train_args),
            "epoch": 0,
            "global_step": 0,
            "val_metrics": {},
        },
        out_path,
    )
    print(f"Saved temporary checkpoint: {out_path}")


if __name__ == "__main__":
    main()
