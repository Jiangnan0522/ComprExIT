import os
from overrides import overrides
from abc import ABC, abstractmethod
from typing import Callable
from tqdm import trange

import torch
from loguru import logger
import evaluate
from datasets import load_dataset


class Metrics(ABC):
    name = 'base'
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def compute(self, predictions:list[str], references:list[str])->dict:
        """
        Args:
            predictions: list of generated texts
            references: list of ground truth texts

            The order of predictions should match the order of references.

        Returns:
            A dictionary of {metric_name: metric_value}.
        """
        pass


class EM_F1(Metrics):
    name = 'em_f1'
    def __init__(self):
        self.metric_name = 'squad'
        self.metric = evaluate.load(self.metric_name)

    def _formatting(self, predictions:list[str], references:list[str]):
        predictions_all = []
        references_all = []
        for sample_id, (pred, ref) in enumerate(zip(predictions, references)):
            predictions_all.append({'prediction_text': pred, 'id': str(sample_id)})
            if isinstance(ref, str):
                ref = [ref]
            references_all.append({'answers': {'text': ref, 'answer_start': [-1]}, 'id': str(sample_id)})
        return predictions_all, references_all
    
    def compute(self, predictions:list[str], references:list[str]):
        predictions, references = self._formatting(predictions, references)
        return self.metric.compute(predictions=predictions, references=references)


class ROUGE(Metrics):
    name = 'rouge'
    def __init__(self):
        self.metric_name = 'rouge'
        self.metric = evaluate.load(self.metric_name)

    def compute(self, predictions:list[str], references:list[str])->dict:
        return self.metric.compute(predictions=predictions, references=references)


class METEOR(Metrics):
    name = 'meteor'
    def __init__(self):
        self.metric_name = 'meteor'
        self.metric = evaluate.load(self.metric_name)

    def compute(self, predictions:list[str], references:list[str])->dict:
        return self.metric.compute(predictions=predictions, references=references)


class Accuracy(Metrics):
    name = 'accuracy'

    def __init__(self):
        self.metric_name = 'accuracy'
        self.metric = evaluate.load(self.metric_name)

    def _convert_string_options_to_int(self, predictions:list[str], references:list[str]):
        all_options = sorted(list(set(references)))
        all_options = [option.strip().lower() for option in all_options]
        predictions_int = []
        # Use a sentinel for invalid predictions so they are always counted as wrong.
        # (The evaluate "accuracy" metric is pure equality; it doesn't have a special "wrong" label.)
        invalid_label = -1
        invalid_count = 0
        invalid_examples: list[str] = []
        for pred in predictions:
            pred_norm = (pred or "").strip().lower()
            if pred_norm in all_options:
                predictions_int.append(all_options.index(pred_norm))
            elif pred_norm and pred_norm[0] in all_options:
                predictions_int.append(all_options.index(pred_norm[0]))
            else:
                invalid_count += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(pred_norm)
                predictions_int.append(invalid_label)

        if invalid_count > 0:
            logger.warning(
                f"{invalid_count}/{len(predictions)} predictions were not in the allowed options {all_options}. "
                f"Treating them as incorrect via sentinel label {invalid_label}. "
                f"Examples (normalized): {invalid_examples}"
            )
        references_int = [all_options.index(ref.strip().lower()) for ref in references]
        return predictions_int, references_int

    def compute(self, predictions:list[str], references:list[str]) -> dict:
        predictions_int, references_int = self._convert_string_options_to_int(predictions, references)
        return self.metric.compute(predictions=predictions_int, references=references_int)


class MetricsFactory:
    def __init__(self, metrics:list[str]):
        self.metrics = []
        for metric_class in Metrics.__subclasses__():
            if metric_class.name in metrics:
                self.metrics.append(metric_class())

    def compute(self, predictions:list[str], references:list[str])->dict:
        results = {}
        for metric in self.metrics:
            res = metric.compute(predictions, references)
            results.update(res)
        return results