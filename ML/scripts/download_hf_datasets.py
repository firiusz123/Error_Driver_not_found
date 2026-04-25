#!/usr/bin/env python3
import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos",
        nargs="+",
        default=[
            "n7i5x9/driver-drowsiness-dataset",
            "akahana/Driver-Drowsiness-Dataset",
        ],
    )
    parser.add_argument("--out-dir", default="data/raw")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for repo_id in args.repos:
        local_dir = os.path.join(args.out_dir, repo_id.replace("/", "__"))
        print(f"downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)


if __name__ == "__main__":
    main()
