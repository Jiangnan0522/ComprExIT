from abc import ABC, abstractmethod
from typing import List
import os
from glob import glob
import datasets as ds
from loguru import logger

class NTPDataset(ABC):
    @abstractmethod
    def load_data(self, data_folder: str, streaming: bool = True, skip_train_files: int = 0):
        pass

class SlimPajamaData6BNTPDataset(NTPDataset):
    name = 'slim_pajama_6b'
    columns_to_remove = ['meta', '__index_level_0__']

    @classmethod
    def _sort_files(cls, files: List[str]) -> List[str]:
        """Sort data files by their shard index (e.g., train-00003-of-00048)."""
        def _extract_index(path: str) -> int:
            # Filename patterns:
            #   "train-00003-of-00048.parquet"
            #   "slim_pajama-6_b-train-00003-of-00048.arrow"
            #   "slim_pajama-6_b-validation.arrow" (single-shard, no index)
            basename = os.path.basename(path)
            for split in ['train', 'validation', 'test']:
                if split in basename:
                    parts = basename.split(split + '-')
                    if len(parts) < 2 or not parts[1]:
                        return 0  # Single-shard file (no index suffix)
                    try:
                        return int(parts[1].split('-')[0])
                    except ValueError:
                        return 0
            return 0
        return sorted(files, key=_extract_index)

    @classmethod
    def _discover_arrow_files(cls, data_folder: str):
        """Discover Arrow files from HF cache directory structure.

        HF stores cached Arrow files at:
            {data_folder}/{dataset_id}/default/0.0.0/{hash}/*.arrow
        """
        arrow_train = cls._sort_files(glob(os.path.join(data_folder, '**/*train*.arrow'), recursive=True))
        arrow_val = cls._sort_files(glob(os.path.join(data_folder, '**/*validation*.arrow'), recursive=True))
        arrow_test = cls._sort_files(glob(os.path.join(data_folder, '**/*test*.arrow'), recursive=True))
        return arrow_train, arrow_val, arrow_test

    @classmethod
    def _discover_parquet_files(cls, data_folder: str):
        """Discover parquet files from a data/ subdirectory."""
        if os.path.basename(os.path.dirname(data_folder)) != 'data':
            data_folder = os.path.join(data_folder, 'data')
        train = cls._sort_files(glob(os.path.join(data_folder, 'train-*')))
        val = cls._sort_files(glob(os.path.join(data_folder, 'validation-*')))
        test = cls._sort_files(glob(os.path.join(data_folder, 'test-*')))
        return train, val, test

    @classmethod
    def load_data(cls, data_folder: str, streaming: bool = True, skip_train_files: int = 0):

        # When not skipping files, use default HF loading (fast path)
        if skip_train_files <= 0:
            try:
                return ds.load_dataset(data_folder, streaming=streaming)
            except Exception:
                logger.info(f"Failed to load dataset from {data_folder} using ds.load_dataset. Trying explicit file listing.")

        # Explicit file listing: try Arrow files first, then parquet
        train_files, val_files, test_files = cls._discover_arrow_files(data_folder)
        file_format = 'arrow'

        if not train_files:
            train_files, val_files, test_files = cls._discover_parquet_files(data_folder)
            file_format = 'parquet'

        if not train_files:
            raise FileNotFoundError(
                f"No Arrow or parquet train files found under {data_folder}. "
                "Check that the dataset exists and has been downloaded."
            )

        # Skip the first N train files
        if skip_train_files > 0:
            if skip_train_files >= len(train_files):
                raise ValueError(
                    f"skip_train_files={skip_train_files} >= total train files={len(train_files)}. "
                    "Cannot skip all training data."
                )
            skipped = train_files[:skip_train_files]
            train_files = train_files[skip_train_files:]
            logger.info(
                f"[Data loading] Skipped {len(skipped)} train files "
                f"({os.path.basename(skipped[0])} ... {os.path.basename(skipped[-1])})"
            )

        file_paths = {'train': train_files, 'validation': val_files, 'test': test_files}
        # Remove empty splits
        file_paths = {k: v for k, v in file_paths.items() if v}

        for split, files in file_paths.items():
            logger.info(f"[Data loading] {split}: {len(files)} {file_format} files")

        dataset = ds.load_dataset(file_format, data_files=file_paths, streaming=streaming)
        return dataset
