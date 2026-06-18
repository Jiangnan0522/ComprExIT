import os
from pathlib import Path
from huggingface_hub import snapshot_download
from argparse import ArgumentParser

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--repo_id", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    target_dir = Path(__file__).resolve().parent
    token = os.environ.get("HUGGINGFACE_TOKEN")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.target_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
if __name__ == "__main__":
    main()
