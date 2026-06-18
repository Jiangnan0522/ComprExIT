import os
from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to download dataset to")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--name", type=str, default=None, help="Dataset subset name")
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # Download dataset to specified directory
    dataset = load_dataset(
        args.repo_id,
        name=args.name,
        cache_dir=args.target_dir,
        token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    
    print(f"Dataset downloaded to: {args.target_dir}")
    print(f"Dataset info: {dataset}")
if __name__ == "__main__":
    main()

