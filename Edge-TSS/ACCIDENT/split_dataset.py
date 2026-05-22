import argparse
import shutil
from pathlib import Path
import pandas as pd

def stratified_split(metadata: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for _, group in metadata.groupby("type", sort=True):
        train = group.sample(frac=train_ratio, random_state=seed)
        test = group.drop(train.index)
        train_parts.append(train)
        test_parts.append(test)
    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df

def copy_videos(rows: pd.DataFrame, dataset_root: Path, target_dir: Path) -> pd.DataFrame:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = rows.copy()
    new_paths = []
    for path in copied["path"].astype(str):
        src = dataset_root / path
        if not src.is_file():
            raise FileNotFoundError(f"Missing source video: {src}")
        dst = target_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        new_paths.append(f"{target_dir.name}/{src.name}")
    copied["path"] = new_paths
    return copied

def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def print_counts(name: str, metadata: pd.DataFrame) -> None:
    print(f"\n{name}: {len(metadata)} videos")
    print(metadata["type"].value_counts().sort_index().to_string())

def main() -> None:
    parser = argparse.ArgumentParser(description="Split real ACCIDENT videos into stratified train/test sets.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata-real.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    metadata = pd.read_csv(args.metadata)
    train_df, test_df = stratified_split(metadata=metadata, train_ratio=args.train_ratio, seed=args.seed)
    train_dir = dataset_root / "train_real_videos"
    test_dir = dataset_root / "test_real_videos"
    recreate_dir(train_dir)
    recreate_dir(test_dir)
    train_df = copy_videos(rows=train_df, dataset_root=dataset_root, target_dir=train_dir)
    test_df = copy_videos(rows=test_df, dataset_root=dataset_root, target_dir=test_dir)
    train_path = dataset_root / "metadata-train-real.csv"
    test_path = dataset_root / "metadata-test-real.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print_counts("Train", train_df)
    print_counts("Test", test_df)
    print(f"\nSaved {train_path}")
    print(f"Saved {test_path}")

if __name__ == "__main__":
    main()