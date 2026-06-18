from abc import ABC, abstractmethod
from glob import glob
import os
import datasets as ds
from typing import List
from loguru import logger
from datasets import DatasetDict

from .sft_datasets import SFTDatasetBase, get_sft_dataset_class_factory
from .ntp_datasets import NTPDataset

def data_loading_factory(name: str, data_folder: str, streaming: bool = True, skip_train_files: int = 0):
    "Return the data loading class for loading."
    for dataset_class in NTPDataset.__subclasses__():
        if dataset_class.name == name:
            columns_to_remove = dataset_class.columns_to_remove
            raw_datasets = dataset_class.load_data(data_folder=data_folder, streaming=streaming, skip_train_files=skip_train_files)
            return raw_datasets, columns_to_remove
    raise KeyError(f"No data loading class found for name: {name}")


def sft_data_loading_factory(dataset_names:list[str], datasets_dir:str) -> DatasetDict:
    "Return the *loaded* SFT dataset for use."
    dataset_dicts: list[DatasetDict] = []
    for dataset_name in dataset_names:
        dataset_class = get_sft_dataset_class_factory(dataset_name)
        logger.info(f"Loading dataset: {dataset_class.name}")
        dataset_dicts.append(dataset_class(datasets_dir).load_dataset())

    if not dataset_dicts:
        raise ValueError(f"No datasets found for names: {dataset_names}")

    # Concatenate per split (train/validation/test/...)
    split_to_datasets: dict[str, list[ds.Dataset]] = {}
    for dsd in dataset_dicts:
        for split, dset in dsd.items():
            split_to_datasets.setdefault(split, []).append(dset)

    concatenated = DatasetDict(
        {split: ds.concatenate_datasets(dsets) for split, dsets in split_to_datasets.items()}
    )
    columns_to_remove = []
    return concatenated, columns_to_remove




__all__ = ["DataLoading", "data_loading_factory", "sft_data_loading_factory"]

if __name__ == "__main__":
    dataset_names = ['squad', 'hotpot_qa']
    datasets_dir = os.environ.get('DATA_DIR', '/path/to/datasets')

    raw_datasets, columns_to_remove = sft_data_loading_factory(dataset_names=dataset_names, datasets_dir=datasets_dir)

    for split, dataset in raw_datasets.items():
        print(f'split: {split}, dataset size: {len(dataset)}')
