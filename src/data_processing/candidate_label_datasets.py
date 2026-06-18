import os
import random
from abc import ABC, abstractmethod

from datasets import load_from_disk, load_dataset, Dataset, DatasetDict


class CandidateLabelDatasetBase(ABC):
    """Base class for classification datasets with a fixed set of candidate labels.

    Subclasses must define `name`, `candidate_labels`, `metric_names`, and implement `load_dataset()`.
    The returned DatasetDict must have at least `claim` and `label` columns. Optionally `context`.
    """

    name: str = 'base'
    candidate_labels: list[str] = []
    metric_names: list[str] = ['accuracy']
    question: str = ''  # Instruction telling the model what to do
    answer_prompt: str = '\nAnswer:'  # Suffix appended after the question, model predicts next token

    def __init__(self, datasets_dir: str, **kwargs):
        self.datasets_dir = datasets_dir

    @abstractmethod
    def load_dataset(self) -> DatasetDict:
        pass

    def format_options_text(self) -> str:
        return "\n".join(f"{chr(65 + i)}) {label}" for i, label in enumerate(self.candidate_labels))

    def format_question(self, example: dict) -> str:
        """Format a single example into the question text (everything except context).

        Subclasses should override this for dataset-specific formatting.
        The returned text is placed after the compressed context and before "Answer:".
        """
        return f"{self.question}\n{self.format_options_text()}"

    def label_to_index(self, label: str) -> int:
        return self.candidate_labels.index(label)

    def index_to_label(self, index: int) -> str:
        return self.candidate_labels[index]


class FEVERDataset(CandidateLabelDatasetBase):
    """FEVER fact verification dataset with gold evidence context.

    Uses the preprocessed v1.0_with_evidence version where each unique claim
    has its gold evidence sentences joined into a `context` column.
    Claims with label NOT ENOUGH INFO have empty context.
    """

    name = 'fever'
    candidate_labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    metric_names = ['accuracy']
    question = 'Based on the evidence, is the claim supported, refuted, or is there not enough information?'

    def format_question(self, example: dict) -> str:
        return f"Claim: {example['claim']}\nQuestion: {self.question}\n{self.format_options_text()}"

    def load_dataset(self) -> DatasetDict:
        path = os.path.join(self.datasets_dir, 'fever', 'v1.0_with_evidence')
        return load_from_disk(path)


class SST2Dataset(CandidateLabelDatasetBase):
    """SST-2 binary sentiment classification for In-Context Learning evaluation.

    Few-shot demonstrations are sampled from the train split and used as the
    compressed context. The test sentence is the uncompressed query.
    """

    name = 'sst2'
    candidate_labels = ['negative', 'positive']
    metric_names = ['accuracy']
    question = 'What is the sentiment of the following sentence?'

    LABEL_MAP = {0: 'negative', 1: 'positive'}

    def __init__(self, datasets_dir: str, num_shots: int = 16, seed: int = 42, **kwargs):
        super().__init__(datasets_dir)
        self.num_shots = num_shots
        self.seed = seed

    def _sample_few_shot_examples(self, train_split) -> list[dict]:
        """Sample balanced few-shot examples (equal positive/negative) from training data."""
        rng = random.Random(self.seed)

        positives = [i for i in range(len(train_split)) if train_split[i]['label'] == 1]
        negatives = [i for i in range(len(train_split)) if train_split[i]['label'] == 0]

        n_per_class = self.num_shots // 2
        selected_idx = rng.sample(positives, n_per_class) + rng.sample(negatives, self.num_shots - n_per_class)
        rng.shuffle(selected_idx)

        return [train_split[i] for i in selected_idx]

    def _format_few_shot_context(self, examples: list[dict]) -> str:
        """Format few-shot examples into a demonstration context string."""
        parts = []
        for ex in examples:
            label_str = self.LABEL_MAP[ex['label']]
            parts.append(f'Sentence: "{ex["sentence"]}"\nSentiment: {label_str}')
        return '\n\n'.join(parts)

    def format_question(self, example: dict) -> str:
        return f'Sentence: "{example["claim"]}"\n{self.question}\n{self.format_options_text()}'

    def load_dataset(self) -> DatasetDict:
        ds = load_dataset('stanfordnlp/sst2')

        few_shot_examples = self._sample_few_shot_examples(ds['train'])
        context_str = self._format_few_shot_context(few_shot_examples)

        # Build from dict to avoid ClassLabel auto-casting strings back to ints
        result = DatasetDict()
        for split_name in ['validation']:
            split = ds[split_name]
            result[split_name] = Dataset.from_dict({
                'claim': [ex['sentence'] for ex in split],
                'label': [self.LABEL_MAP[ex['label']] for ex in split],
                'context': [context_str] * len(split),
            })

        return result


def get_candidate_label_dataset_factory(name: str) -> type:
    for subclass in CandidateLabelDatasetBase.__subclasses__():
        if getattr(subclass, 'name', None) == name:
            return subclass
    raise ValueError(f"Unknown candidate label dataset: {name}. "
                     f"Available: {[c.name for c in CandidateLabelDatasetBase.__subclasses__()]}")
