import json
import re
import os
from overrides import overrides
from abc import ABC, abstractmethod
from typing import Callable
from tqdm import trange

import pandas as pd
import torch
from loguru import logger
import evaluate
import datasets as ds
from datasets import load_dataset, DatasetDict

from src.evaluation.metrics import MetricsFactory
from src.model.inference import compressing_predict_with_question_and_context, beacon_predict, base_model_predict, base_model_predict_single, base_model_predict_batch
from src.data_processing.sft_datasets import SFTDatasetBase
from src.data_processing.prompte_template import QATemplate


class Evaluator:
    """Evaluate a given dataset
    """

    def __init__(
        self, 
        dataset:SFTDatasetBase,
        split:str='validation',
        max_new_tokens=20,
        max_question_length=64,
        max_context_length=512,
        ):
        
        self.template = QATemplate()

        self.dataset_raw = dataset
        self.dataset = self.dataset_raw.load_dataset(keep_single_answer=False)[split] # keep the list of answers
        self.metrics = MetricsFactory(self.dataset_raw.metric_names)
        self.max_new_tokens = max_new_tokens
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length

    @property
    @abstractmethod
    def metric_names(self) -> list[str]:
        return self.dataset_raw.metric_names
    
    @property
    @abstractmethod
    def name(self) -> str:
        return self.dataset_raw.name


    def _format_result_string(self, results:dict) -> str:
            result_str = ''
            for metric_name, metric_value in results.items():
                result_str += f'{metric_name}: {metric_value:.3f}\n'
            return result_str

    def get_record(self, pred_text:str, example:dict) -> dict:
        """
        Get the record of a single sample of generated text and one corresponding example.

        Returns:
            record: a customised dictionary. Can be: {question, context, pred, gold_clean, gold}
        """
        return {
            'question': example['question'],
            'context': example['context'],
            'pred': pred_text,
            'gold': example['gold']
        }

    def get_batch_questions_and_contexts(self, examples:list[dict]) -> tuple[list[str], list[str]]:
        """
        Get the batch of questions, contexts, and gold answers from the examples.

        Returns:
            batch_questions: list of questions
            batch_contexts: list of contexts
            batch_golds: list of gold answers
        """
        batch_contexts = [self.template.build_context_text(example['context']) for example in examples]
        batch_questions = [self.template.build_question_text(example['question']) for example in examples]
        batch_golds = [example['gold'] for example in examples] # no stringification needed
        return batch_questions, batch_contexts, batch_golds


    @torch.no_grad()
    def evaluate(
            self, 
            model, 
            tokenizer, 
            device, 
            batch_size=8, 
            compress=True, 
            with_context=True, 
            compress_ratio=4,
            predict_func=compressing_predict_with_question_and_context
        ):

        predictions_all = []
        references_all = []
        records_all = []
        for i in trange(0, len(self.dataset), batch_size, desc=f'Evaluating on {self.name}.'):
            end_idx = min(i+batch_size, len(self.dataset))
            examples = self.dataset.select(range(i, end_idx))

            # predict
            batch_questions, batch_contexts, batch_golds = self.get_batch_questions_and_contexts(examples)
            prompt = self.template.prompt_text

            pred_texts = predict_func(
                questions=batch_questions,
                contexts=batch_contexts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                answer_extractor=self.dataset_raw.extract_answer,
                max_new_tokens=self.max_new_tokens,
                max_question_length=self.max_question_length,
                max_context_length=self.max_context_length,
                # Control ablations
                compress=compress,
                with_context=with_context,
                compress_ratio=compress_ratio,
                # Gold answers for oracle selection
                gold_answers=batch_golds,
                # Original records for oracle char_span mode (stored as JSON strings)
                orig_records=[json.loads(ex['orig_record']) for ex in examples] if 'orig_record' in examples.column_names else None,
                )

            for pred_text, gold, example in zip(pred_texts, batch_golds, examples):
                record = self.get_record(pred_text, example)
                records_all.append(record)
                predictions_all.append(pred_text)
                references_all.append(gold)

        result_dict = self.metrics.compute(predictions=predictions_all, references=references_all)
        result_str = self._format_result_string(result_dict)
        return result_str, records_all


def _resolve_max_context_length(model, cli_override, fallback: int = 512) -> int:
    """Pick the eval-time context budget.

    Priority: CLI override > model.config.context_length > fallback.
    """
    if cli_override is not None:
        return int(cli_override)
    cfg = getattr(model, "config", None)
    val = getattr(cfg, "context_length", None)
    if val is None:
        logger.warning(
            f"Model config has no 'context_length'; falling back to {fallback}. "
            "Pass --max_context_length explicitly to avoid this."
        )
        return fallback
    return int(val)


if __name__ == '__main__':
    import json
    import shutil
    from tqdm import tqdm
    import traceback
    from pathlib import Path
    import gc # Added for garbage collection

    from src.device_utils import get_device_module
    from src.data_processing.sft_datasets import get_sft_dataset_class_factory
    from src.evaluation.utils_eval import get_model_and_tokenizer

    DEVICE_MODULE, DEVICE_TYPE = get_device_module()

    SHORT_DATASET_NAMES = ['squad', 'hotpot_qa', 'race', 'adversarial_qa', 'drop']
    LONG_DATASET_NAMES = ["triviaqa_wikipedia", "quality", "natural_questions"]
    MRQA_DATASET_NAMES = ['mrqa_natural_questions', 'mrqa_trivia_qa', 'mrqa_news_qa', 'mrqa_search_qa', 'mrqa_squad', 'mrqa_hotpot_qa']
    MRQA_OOD_DATASET_NAMES = ['mrqa_bioasq', 'mrqa_drop', 'mrqa_duorc', 'mrqa_race', 'mrqa_relationextraction', 'mrqa_textbookqa']
    SUMMARIZATION_DATASET_NAMES = ['xsum']

    # --------- Args --------------------------
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate models on SFT datasets.')
    parser.add_argument('--datasets_dir', type=str, default='../datasets')
    parser.add_argument('--partition', type=str, default='mrqa', choices=['short', 'long', 'mrqa', 'mrqa_ood', 'summarize'])
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--compress_ratio', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--rerun_and_overwrite', action='store_true', default=False)
    parser.add_argument('--attn_implementation', type=str, default='eager')
    parser.add_argument(
        '--max_context_length', type=int, default=None,
        help="Override max context tokens at eval time. "
             "If omitted, read from each model's config.json (context_length); "
             "falls back to 512 if neither is available."
    )
    parser.add_argument(
        '--model_folders', type=str, nargs='+', required=True,
        help="One or more trained checkpoint directories to evaluate."
    )
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    partition = args.partition
    output_dir = args.output_dir
    compress_ratio = args.compress_ratio
    rerun_and_overwrite = args.rerun_and_overwrite
    attn_implementation = args.attn_implementation
    model_folders = args.model_folders

    # --------- End of Args ---------

    if DEVICE_TYPE == 'xpu':
        DEVICE='xpu:0'
    elif DEVICE_TYPE == 'cuda':
        DEVICE='cuda:0'
    else:
        DEVICE='cpu'
    logger.info(f'Using device: {DEVICE}')

    logger.add(os.path.join(output_dir, 'evaluation_results.log'))

    # Select datasets based on partition
    if partition == 'short':
        dataset_names = SHORT_DATASET_NAMES
        logger.info(f'Using short context datasets: {dataset_names}')
    elif partition == 'long':
        dataset_names = LONG_DATASET_NAMES
        logger.info(f'Using long context datasets: {dataset_names}')
    elif partition == 'mrqa':
        dataset_names = MRQA_DATASET_NAMES
        logger.info(f'Using MRQA datasets: {dataset_names}')
    elif partition == 'mrqa_ood':
        dataset_names = MRQA_OOD_DATASET_NAMES
        logger.info(f'Using MRQA OOD datasets: {dataset_names}')
    elif partition == 'summarize':
        dataset_names = SUMMARIZATION_DATASET_NAMES
        logger.info(f'Using summarization datasets: {dataset_names}')
    else:
        raise ValueError(f'Invalid partition: {partition}')

    not_found_models = []
    pbar = tqdm(model_folders)
    for model_folder in pbar:
        # create output directory for the model        
        logger.info(f'Loading model from {model_folder}...')
        
        try:
            if Path(model_folder).name.startswith('checkpoint'):
                # if the model folder is a checkpoint
                model_parent = Path(model_folder).parent.name
                checkpoint_name = Path(model_folder).name
                unique_model_name = f"{model_parent}_{checkpoint_name}"
            else:
                checkpoint_name = Path(model_folder).name
                unique_model_name = checkpoint_name

            with_context = True
            compress = True
            # Baseline detection
            if 'baseline' in checkpoint_name.lower():
                compress = False
                if 'with-context' in checkpoint_name.lower():
                    with_context = True
                else:
                    with_context = False
                unique_model_name = checkpoint_name # use checkpoint name as unique model name for baselines
                model_folder = str(Path(model_folder).parent) # use the parent directory as the model folder for baselines
                logger.info(f'Baseline detected! Running baseline: {unique_model_name}. Compress: {compress}. With context: {with_context}.')

            for dataset_name in dataset_names:
                # create output directory for the dataset
                # FIX: Use unique_model_name
                output_dir_dataset = os.path.join(output_dir, dataset_name, unique_model_name)
                
                # FIX: Logic to skip or overwrite
                if os.path.exists(output_dir_dataset) and (len(os.listdir(output_dir_dataset)) > 0):
                    if not rerun_and_overwrite:
                        logger.info(f'Output directory {output_dir_dataset} already exists. Skipping...')
                        continue
                    else:
                        logger.warning(f'Rerunning and overwriting output directory {output_dir_dataset}...')
                else:
                    os.makedirs(output_dir_dataset, exist_ok=True)

                if not os.path.exists(model_folder):
                    logger.warning(f'Model folder {model_folder} does not exist. Skipping.')
                    not_found_models.append(model_folder)
                    continue
                
                model, tokenizer = get_model_and_tokenizer(
                    model_folder,
                    device=DEVICE,
                    attn_implementation=attn_implementation,
                )
                dataset_class = get_sft_dataset_class_factory(dataset_name)
                dataset = dataset_class(datasets_dir)

                max_new_tokens = 32 if partition == 'summarize' else 20
                max_context_length = _resolve_max_context_length(model, args.max_context_length)
                logger.info(
                    f'Using max_context_length={max_context_length} for '
                    f'{unique_model_name} on {dataset_name} '
                    f'(cli_override={args.max_context_length}, '
                    f'model_config={getattr(getattr(model, "config", None), "context_length", None)}).'
                )
                evaluator = Evaluator(
                    dataset,
                    max_new_tokens=max_new_tokens,
                    max_context_length=max_context_length,
                )

                predict_func = compressing_predict_with_question_and_context # default encoder-decoder prediction function
                # Special case: beacon model - since beacon is single-llm structure, use it as a no-compression model
                if 'beacon' in model_folder:
                    with_context = True
                    compress = False
                    logger.info(f'Beacon model detected! Setting - compress: {compress}. with_context: {with_context}.')
                    predict_func = beacon_predict
                elif 'prompt_tuning' in unique_model_name.lower():
                    compress = False
                    with_context = True
                    logger.info(f'Prompt tuning detected! Setting - compress: {compress}. with_context: {with_context}.')
                    predict_func = base_model_predict_batch

                # if partition == 'long' and 'baseline' not in unique_model_name.lower():
                #     model.enable_map_reduce()
                #     logger.info(f'Enabling map-reduce for long context datasets...')
                # else:
                #     model.disable_map_reduce()

                pbar.set_description(f'Evaluating {unique_model_name} on {dataset_name}.')
                result_str, records_all = evaluator.evaluate(
                    model, 
                    tokenizer, 
                    DEVICE, 
                    batch_size=args.batch_size,
                    compress_ratio=compress_ratio, 
                    with_context=with_context, 
                    compress=compress,
                    predict_func=predict_func,
                )

                # save results
                with open(os.path.join(output_dir_dataset, 'results.txt'), 'w') as f:
                    f.write(result_str)
                with open(os.path.join(output_dir_dataset, 'records.json'), 'w') as f:
                    json.dump(records_all, f, indent=4)

                logger.info(f'Dataset: {dataset_name}.')
                logger.info(f'Model: {unique_model_name}.')
                logger.info(result_str)
                logger.info('-'*100)

        except Exception as e:
            logger.error(f'Error evaluating {unique_model_name} on {dataset_name}. Skipped.')
            logger.error(traceback.format_exc())
            logger.info('-'*100)
            continue
        
        finally:
            logger.info(f'Cleaning up resources for model {unique_model_name}...')
            logger.info(f'Not found models: {not_found_models}')
            # Resource cleanup to prevent OOM in loop
            if 'model' in locals():
                del model
            torch.cuda.empty_cache() # Use torch.xpu.empty_cache() if available on your system
            gc.collect()