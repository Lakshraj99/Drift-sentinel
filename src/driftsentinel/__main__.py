"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .figures import generate_figures
from .pipeline import run_pipeline
from .utils import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="driftsentinel")
    parser.add_argument("command", choices=("train", "evaluate", "reproduce"), nargs="?", default="reproduce")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    configure_logging()
    root = Path.cwd()
    config = load_config(root / args.config)
    if args.command == "evaluate":
        generate_figures(root, config, int(config["seeds"][0]))
    else:
        run_pipeline(config, root, quick=args.quick)


if __name__ == "__main__":
    main()
