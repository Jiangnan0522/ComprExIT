from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
import os
import gzip
import json
import datasets as ds
from datasets import Dataset, DatasetDict
import pandas as pd


class SFTDatasetBase(ABC):
    name = 'base'

    def __init__(self, datasets_dir:str, **kwargs: Any):
        self.datasets_dir = datasets_dir
        self.kwargs = kwargs

    @abstractmethod
    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        """
        Load the dataset from the dataset folder, and process it to the standard SFT schema.

        Schema:
                - record_format:
                    {
                        "context": str,
                        "question": str,
                        "gold": str | list[str],
                    }
                - splits:
                    must: 'train' and 'validation' splits.
                    optional: 'test' split.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Dataset name identifier.
        """
        pass

    @property
    @abstractmethod
    def metric_names(self) -> list[str]:
        """
        List of metric names to compute.
        """
        pass

    def _resolve_dataset_folder_name(self) -> str:
        """
        Resolve which on-disk folder name to load from.

        Some SFT datasets expose a different public `name` than their underlying
        dataset folder (e.g. `hotpot_qa_long` loads from `hotpot_qa`).
        """
        return getattr(self, "dataset_name", self.name)

    def _load_hf_dataset_from_dir(self) -> DatasetDict:
        dataset_folder = self._resolve_dataset_folder_name()
        dataset_path = os.path.join(self.datasets_dir, dataset_folder)
        dataset = ds.load_dataset(dataset_path)
        return self.correct_split_names(dataset)

    def _normalize_gold_value(self, gold: Any, keep_single_answer: bool) -> Any:
        """
        Some datasets provide multiple gold answers (list). This helper optionally
        converts them into a single supervision target by picking the first.
        """
        if isinstance(gold, list):
            if keep_single_answer:
                return gold[0] if gold else ""
            return gold
        return gold

    def _map_to_sft_columns(
        self,
        dataset: DatasetDict,
        format_fn: Callable[[dict], dict],
        keep_single_answer: bool = True,
        load_from_cache_file: bool = True,
    ) -> DatasetDict:
        """
        Convert each split to the standard SFT schema:
            { "context": str, "question": str, "gold": str | list[str] }

        We remove original columns to keep downstream preprocessing lightweight.
        """
        def _wrapped_format_fn(example: dict) -> dict:
            out = format_fn(example)
            if "gold" in out:
                out["gold"] = self._normalize_gold_value(out["gold"], keep_single_answer=keep_single_answer)
            return out

        mapped: dict[str, Dataset] = {}
        for split, dset in dataset.items():
            mapped[split] = dset.map(
                _wrapped_format_fn,
                load_from_cache_file=load_from_cache_file,
                remove_columns=list(dset.column_names),
                desc=f"Formatting SFT dataset ({self.name}/{split})",
            )
        return DatasetDict(mapped)

    def correct_split_names(self, dataset: DatasetDict) -> DatasetDict:
        for split in list(dataset.keys()):  # snapshot keys
            if split != "validation" and (("val" in split) or (split in {"dev", "valid"})):
                if "validation" in dataset and split != "validation":
                    raise ValueError(f"Both 'validation' and '{split}' splits exist.")
                dataset["validation"] = dataset.pop(split)
        return dataset

    def get_stats(self, dataset: DatasetDict) -> dict:
        # get maximum length of context and question
        max_context_length = 0
        max_question_length = 0
        max_gold_length = 0

        for split in dataset.keys():
            for example in dataset[split]:
                context_length = len(example['context'].split())
                question_length = len(example['question'].split())
                gold = example["gold"]
                if isinstance(gold, list):
                    gold_length = max((len(str(a).split()) for a in gold), default=0)
                else:
                    gold_length = len(str(gold).split())

                max_context_length = max(max_context_length, context_length)
                max_question_length = max(max_question_length, question_length)
                max_gold_length = max(max_gold_length, gold_length)

        return {
            'max_context_words': max_context_length,
            'max_question_words': max_question_length,
            'max_gold_words': max_gold_length
        }


    @staticmethod
    def _find_answer_spans(context: str, answers: list[str]) -> list[list[int]]:
        """Find character spans [start, end) of answer strings in context via string search.

        Returns deduplicated spans. If an answer is not found in context, it is skipped.
        """
        spans = []
        seen: set[tuple[int, int]] = set()
        for answer in answers:
            start = context.find(answer)
            if start != -1:
                key = (start, start + len(answer))
                if key not in seen:
                    seen.add(key)
                    spans.append([key[0], key[1]])
        return spans

    def extract_answer(self, text:str) -> str:
        """
        Extract the answer from a single sample of generated text for QA datasets.
        """
        return text.split('.')[0].split('\n')[0].strip()


class SquadSFTDataset(SFTDatasetBase):
    name = 'squad'
    metric_names = ['em_f1', 'rouge', 'meteor']
    def __init__(self, datasets_dir:str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def format_dataset(self, example: dict) -> dict:
        gold_span = [
            [s, s + len(t)]
            for s, t in zip(example['answers']['answer_start'], example['answers']['text'])
        ]
        return {
            "context": example['context'],
            "question": example['question'],
            # SQuAD provides multiple acceptable answers; keep as list and normalize in load_dataset().
            "gold": example['answers']['text'],
            "gold_span": gold_span,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")

        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class HotpotQASFTDataset(SFTDatasetBase):
    name = 'hotpot_qa'
    metric_names = ['em_f1', 'rouge', 'meteor']
    def __init__(self, datasets_dir:str, use_gold_context:bool=True, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)
        self.use_gold_context = use_gold_context

    def _get_contexts(self, example:dict) -> str:
        gold_titles = set(example['supporting_facts']['title'])
        titles = example['context']['title']
        articles = example['context']['sentences']

        if self.use_gold_context:
            ## Get contexts of gold articles only
            context_per_question = ''
            for gold_title in gold_titles:
                idx = titles.index(gold_title)
                gold_article_sentences = articles[idx]
                context_per_question += ' '.join(gold_article_sentences)
        else:
            ## Get contexts of all articles
            context_per_question = ''
            for article in articles:
                context_per_question += ' '.join(article)
        return context_per_question

    def format_dataset(self, example: dict) -> dict:
        context = self._get_contexts(example)
        answer = example['answer']
        gold_span = self._find_answer_spans(context, [answer])
        return {
            "context": context,
            "question": example['question'],
            "gold": answer,
            "gold_span": gold_span,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")

        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class HotpotQALongSFTDataset(HotpotQASFTDataset):
    """
    Long-context variant of HotpotQA.

    This exposes `name='hotpot_qa_long'` but loads from the underlying `hotpot_qa`
    dataset folder and uses *all* articles as context.
    """

    name = "hotpot_qa_long"
    metric_names = ['em_f1', 'rouge', 'meteor']
    dataset_name = "hotpot_qa"

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir=datasets_dir, use_gold_context=False, **kwargs)


class RACEsFTDataset(SFTDatasetBase):
    name = "race"
    metric_names = ['accuracy']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _format_question_with_options(self, example: dict) -> str:
        q = example["question"]
        opts = example["options"]
        # Keep question text *without* "Question:" prefix; preprocessing adds it.
        question_string = f"{q}. Options:"
        for idx, opt in zip(["A", "B", "C", "D"], opts):
            question_string += f" {idx}.{opt}"
        return question_string

    def format_dataset(self, example: dict) -> dict:
        return {
            "context": example["article"],
            "question": self._format_question_with_options(example),
            "gold": str(example["answer"]).strip(),
            "gold_span": [],  # multiple choice — no extractive span
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class DropSFTDataset(SFTDatasetBase):
    name = "drop"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def format_dataset(self, example: dict) -> dict:
        spans = example.get("answers_spans", {}).get("spans")
        context = example["passage"]
        gold_span = self._find_answer_spans(context, spans) if spans else []
        return {
            "context": context,
            "question": example["question"],
            # DROP can have multiple spans; keep as list and normalize in load_dataset().
            "gold": spans if spans is not None else [],
            "gold_span": gold_span,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class AdversarialQASFTDataset(SFTDatasetBase):
    name = "adversarial_qa"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def format_dataset(self, example: dict) -> dict:
        answers = example.get("answers", {}).get("text", [])
        answer_starts = example.get("answers", {}).get("answer_start", [])
        gold_span = [
            [s, s + len(t)]
            for s, t in zip(answer_starts, answers if isinstance(answers, list) else [answers])
        ]
        return {
            "context": example["context"],
            "question": example["question"],
            # Multiple answers exist; keep as list and normalize in load_dataset().
            "gold": answers if isinstance(answers, list) else ([answers] if answers else []),
            "gold_span": gold_span,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class NarrativeQASFTDataset(SFTDatasetBase):
    """
    NarrativeQA (summary-context version).

    Expected on-disk structure (same as `EvalDataset` version):
        - {datasets_dir}/narrativeqa/third_party/wikipedia/summaries.csv
        - {datasets_dir}/narrativeqa/qaps.csv

    We use:
        - `summary` as context
        - `answer2` as gold answer
    """

    name = "narrativeqa"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _load_narrativeqa_from_csv(self) -> DatasetDict:
        data_path = os.path.join(self.datasets_dir, self.name)
        df_sums = pd.read_csv(os.path.join(data_path, "third_party", "wikipedia", "summaries.csv"))
        df_questions = pd.read_csv(os.path.join(data_path, "qaps.csv"))

        # Keep only columns we need and normalize names.
        cols_mapping = {
            "document_id": "id",
            "question": "question",
            "answer2": "gold",
            "summary": "context",
        }
        df_dataset = df_questions.merge(df_sums.drop(columns=["set"]), on="document_id", how="left")

        split_to_rows: dict[str, list[dict]] = {}
        for split in df_dataset["set"].unique():
            df_split = df_dataset[df_dataset["set"] == split]
            df_split = df_split.loc[:, cols_mapping.keys()].rename(columns=cols_mapping)

            # Drop rows with missing gold answer.
            is_na = df_split["gold"].isna()
            removed = int(is_na.sum())
            if removed > 0:
                logger.info(
                    f"{self.name}: Number of rows removed due to the absence of answer for summary context: {removed}"
                )
            df_split = df_split[~is_na]

            # Ensure strings
            df_split["context"] = df_split["context"].astype(str)
            df_split["question"] = df_split["question"].astype(str)
            df_split["gold"] = df_split["gold"].astype(str)

            records = df_split.to_dict(orient="records")
            for r in records:
                r["gold_span"] = self._find_answer_spans(r["context"], [r["gold"]])
            split_to_rows[str(split)] = records
            logger.info(f"{self.name}: length of {split} dataset: {len(split_to_rows[str(split)])}")

        dataset = DatasetDict({s: Dataset.from_list(rows) for s, rows in split_to_rows.items()})
        dataset = self.correct_split_names(dataset)
        return dataset

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_narrativeqa_from_csv()
        # Keep schema consistent; gold is already a string here, but run through normalizer anyway.
        dataset = DatasetDict(
            {
                split: dset.map(
                    lambda ex: {
                        "gold": self._normalize_gold_value(ex.get("gold"), keep_single_answer=keep_single_answer)
                    },
                    load_from_cache_file=True,
                    desc=f"Normalizing gold answers ({self.name}/{split})",
                )
                for split, dset in dataset.items()
            }
        )
        # Already in SFT schema; still compute stats for logging consistency.
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


# --- MRQA Family ---
class MRQASFTDatasetBase(SFTDatasetBase):
    name = "mrqa_base"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa"

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def clean_context(self, context:str) -> str:
        return (
            context.replace("[PAR] ", "\n\n")
                .replace("[TLE]", "Title:")
                .replace("[SEP]", "\nPassage:").strip()
                .replace("<Li>", "")
                .replace("</Li>", "")
                .replace("<OI>", "")
                .replace("</OI>", "")
                .replace("<Ol>", "")
                .replace("</Ol>", "")
                .replace("<Dd>", "")
                .replace("</Dd>", "")
                .replace("<UI>", "")
                .replace("</UI>", "")
                .replace("<Ul>", "")
                .replace("</Ul>", "")
                .replace("<P>", "")
                .replace("</P>", "")
                .replace("[DOC]", "")
        ).strip()


    def clean_up_spaces(self, s:str) -> str:
        out_string = s
        return (
            out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
                .replace("( ", "(")
                .replace(" )", ")")
                .replace(" %", "%")
                .replace("`` ", "\"")
                .replace(" ''", "\"")
                .replace(" :", ":")
        )

    def format_dataset(self, example: dict) -> dict:
        context = self.clean_context(example['context'])

        question = example['question'].strip()
        if question[-1] != "?":
            question += "?"
        question = self.clean_up_spaces(question)

        gold = [self.clean_up_spaces(a) for a in example['gold']]

        gold_span = self._find_answer_spans(context, gold)

        out = {
            'context': context,
            'question': question,
            'gold': gold,
            'gold_span': gold_span,
        }
        if 'orig_record' in example:
            out['orig_record'] = example['orig_record']
        return out

    def load_dataset_for_mrqa(self, datasets_dir:str, dataset_name:str) -> DatasetDict:
        """
        Load dataset for MRQA.

        Record format before formatting:
            {
                'context': str,
                'question': str,
                'gold': list[str],
            }
        """
        dataset_name = dataset_name.replace('mrqa_', '')
        dataset_dir = os.path.join(datasets_dir, self.folder_name, dataset_name)

        dataset_dict = {}
        for file in os.listdir(dataset_dir):
            if not file.endswith('.jsonl.gz'):
                continue

            split_name = file.split('_')[0]
            if split_name in dataset_dict:
                raise ValueError(f"Split name {split_name} already exists in dataset_dict, which may indicate duplicate splits.")

            path = os.path.join(dataset_dir, file)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                header = json.loads(next(f))  # first line is the header JSON

                records = []
                for line in f:
                    record_raw = json.loads(line)
                    context = record_raw['context'] # one context with multiple qas
                    for qa in record_raw['qas']:
                        question = qa['question']
                        answers = qa['answers']
                        record = {
                            'context': context,
                            'question': question,
                            'gold': answers,
                            'orig_record': json.dumps(qa)
                        }
                        records.append(record)
                dataset_dict[split_name] = records
        return DatasetDict({split_name: Dataset.from_list(data) for split_name, data in dataset_dict.items()})


    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self.load_dataset_for_mrqa(self.datasets_dir, self.name)
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=False
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class MRQAofSQUADSFTDataset(MRQASFTDatasetBase):
    name = "mrqa_squad"
    metric_names = ['em_f1', 'rouge', 'meteor']

class MRQAofHotpotQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_hotpot_qa"
    metric_names = ['em_f1', 'rouge', 'meteor']

class MRQAofNaturalQuestionsSFTDataset(MRQASFTDatasetBase):
    name = "mrqa_natural_questions"
    metric_names = ['em_f1', 'rouge', 'meteor']

class MRQAofTriviaQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_trivia_qa"
    metric_names = ['em_f1', 'rouge', 'meteor']

class MRQAofNewsQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_news_qa"
    metric_names = ['em_f1', 'rouge', 'meteor']

class MRQAofSearchQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_search_qa"
    metric_names = ['em_f1', 'rouge', 'meteor']

# MRQA OOD datasets
class MRQAofBioQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_bioasq"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

class MRQAofDropQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_drop"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

class MRQAofDuoQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_duorc"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

class MRQAofRaceQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_race"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

class MRQAofRelationExtractionQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_relationextraction"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

class MRQAofTextbookQASFTDataset(MRQASFTDatasetBase):
    name = "mrqa_textbookqa"
    metric_names = ['em_f1', 'rouge', 'meteor']
    folder_name = "mrqa_ood"

def get_sft_dataset_class_factory(dataset_name: str) -> type[SFTDatasetBase]:
    def _iter_all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _iter_all_subclasses(sub)

    for subclass in _iter_all_subclasses(SFTDatasetBase):
        if getattr(subclass, "name", None) == dataset_name:
            return subclass
    raise ValueError(f"Invalid SFT dataset name: {dataset_name}")


class TriviaQAWikipediaSFTDataset(SFTDatasetBase):
    """
    TriviaQA (rc.wikipedia) – extractive QA over long Wikipedia articles.

    Expected on-disk structure:
        {datasets_dir}/triviaqa_wikipedia/   (saved via datasets.save_to_disk)

    Context is built by concatenating all Wikipedia entity-page texts.
    Gold answer is the canonical ``value`` field; aliases are kept as a list.
    Typical context lengths: median ~8k tokens, 70% >= 4k tokens.
    """

    name = "triviaqa_wikipedia"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _load_from_disk(self) -> DatasetDict:
        path = os.path.join(self.datasets_dir, self.name)
        dataset = ds.load_from_disk(path)
        return self.correct_split_names(dataset)

    def format_dataset(self, example: dict) -> dict:
        # Concatenate all Wikipedia entity page texts as context
        wiki_contexts = example.get("entity_pages", {}).get("wiki_context", [])
        context = "\n\n".join(wiki_contexts) if wiki_contexts else ""

        answer_dict = example.get("answer", {})
        # Primary answer
        gold_value = answer_dict.get("value", "")
        # All acceptable aliases (for evaluation)
        aliases = answer_dict.get("aliases", [])
        gold = [gold_value] + [a for a in aliases if a != gold_value] if aliases else [gold_value]

        return {
            "context": context,
            "question": example["question"],
            "gold": gold,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_from_disk()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class QuALITYSFTDataset(SFTDatasetBase):
    """
    QuALITY – multiple-choice QA over long articles (~5-7k tokens).

    Expected on-disk structure:
        {datasets_dir}/quality/   (saved via datasets.save_to_disk)

    The correct option text is used as the gold answer string.
    """

    name = "quality"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _load_from_disk(self) -> DatasetDict:
        path = os.path.join(self.datasets_dir, self.name)
        dataset = ds.load_from_disk(path)
        return self.correct_split_names(dataset)

    def format_dataset(self, example: dict) -> dict:
        options = example["options"]
        answer_idx = example["answer"]
        gold_text = options[answer_idx]

        # Include options in the question for context
        options_str = " ".join(
            [f"{chr(65 + i)}.{opt}" for i, opt in enumerate(options)]
        )
        question = f"{example['question']} Options: {options_str}"

        return {
            "context": example["article"],
            "question": question,
            "gold": gold_text,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_from_disk()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class MuSiQueSFTDataset(SFTDatasetBase):
    """
    MuSiQue (answerable) – multi-hop extractive QA with 2-20 paragraphs.

    Expected on-disk structure:
        {datasets_dir}/musique/   (saved via datasets.save_to_disk)

    Context is built by concatenating all paragraph texts.
    Typical context lengths: median ~2k tokens (shorter than 4k target,
    but useful as supplementary data for multi-hop reasoning).
    """

    name = "musique"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _load_from_disk(self) -> DatasetDict:
        path = os.path.join(self.datasets_dir, self.name)
        dataset = ds.load_from_disk(path)
        return self.correct_split_names(dataset)

    def format_dataset(self, example: dict) -> dict:
        paragraphs = example.get("paragraphs", [])
        context = "\n\n".join(
            [f"{p['title']}: {p['paragraph_text']}" for p in paragraphs]
        )

        gold_value = example["answer"]
        aliases = example.get("answer_aliases", [])
        gold = [gold_value] + [a for a in aliases if a != gold_value] if aliases else [gold_value]

        return {
            "context": context,
            "question": example["question"],
            "gold": gold,
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_from_disk()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class NaturalQuestionsSFTDataset(SFTDatasetBase):
    """
    Google Natural Questions – extractive QA over full Wikipedia articles.

    Expected on-disk structure:
        {datasets_dir}/natural_questions/   (saved via datasets.save_to_disk)

    Already preprocessed to {context, question, gold} by download_nq.py.
    Typical context lengths: median ~6.3k tokens.
    """

    name = "natural_questions"
    metric_names = ['em_f1', 'rouge', 'meteor']

    def __init__(self, datasets_dir: str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def _load_from_disk(self) -> DatasetDict:
        path = os.path.join(self.datasets_dir, self.name)
        dataset = ds.load_from_disk(path)
        return self.correct_split_names(dataset)

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_from_disk()
        logger.info(f"Loaded SFT dataset: {self.name}")
        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset


class CNNDailyMailSFTDataset(SFTDatasetBase):
    name = "cnn_dailiymail"
    metric_names = ['em_f1', 'rouge', 'meteor']

    question_template = "Summarize the article in short sentences."

    def __init__(self, datasets_dir:str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def format_dataset(self, example: dict) -> dict:
        return {
            "context": example['article'],
            "question": self.question_template,
            # SQuAD provides multiple acceptable answers; keep as list and normalize in load_dataset().
            "gold": example['hightlights'],
            "gold_span": [],  # summarization — no extractive span
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")

        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset

    def extract_answer(self, text:str) -> str:
        """
        Extract the answer from a single sample of generated text for QA datasets.
        """
        # TODO
        pass


class XSumSFTDataset(SFTDatasetBase):
    name = "xsum"
    metric_names = ['em_f1', 'rouge', 'meteor']

    question_template = "Summarize the article in short sentences."

    def __init__(self, datasets_dir:str, **kwargs: Any):
        super().__init__(datasets_dir, **kwargs)

    def format_dataset(self, example: dict) -> dict:
        return {
            "context": example['document'],
            "question": self.question_template,
            "gold": example['summary'],
            "gold_span": [],  # summarization — no extractive span
        }

    def load_dataset(self, keep_single_answer: bool = True) -> DatasetDict:
        dataset = self._load_hf_dataset_from_dir()
        dataset = self._map_to_sft_columns(
            dataset, self.format_dataset, keep_single_answer=keep_single_answer, load_from_cache_file=True
        )
        logger.info(f"Loaded SFT dataset: {self.name}")

        stats = self.get_stats(dataset)
        logger.info(f"{self.name} stats:\n{stats}")
        return dataset

    def extract_answer(self, text:str) -> str:
        """
        For summarization, return the full generated text (stripped).
        """
        return text.strip()
