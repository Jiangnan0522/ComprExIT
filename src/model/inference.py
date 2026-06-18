from typing import Callable, Optional
import torch

from src.model import EncoderDecoderCompressorBase
from src.model.modelling_utils import move_padding_to   


@torch.no_grad()
def compressing_predict_with_question_and_context(
        questions:list[str], 
        contexts:list[str], 
        model:EncoderDecoderCompressorBase, 
        tokenizer, 
        device:str, 
        prompt:str,
        answer_extractor:Callable = None,
        max_new_tokens=100,
        max_question_length=128,
        max_context_length=512,
        compress:bool = True,
        with_context:bool = True,
        compress_ratio:int = 4,
        **kwargs,
    ):
    """
    Predict with compressed context gists followed by the question tokens.

    For batch prediction, the questions and context are padded or truncated to 
    the max_question_length and max_context_length.

    No need to worry about the position of padding tokens because the model will handle it.

    Args:
        answer_extractor: a function that extracts the answer from one generated text.
    """

    # Sanity check
    if compress and (not with_context):
        raise ValueError("Compress=True but with_context=False.")

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"   # keep the tail, matching SFT preprocessing
    try:
        inputs_context = tokenizer(
            contexts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_context_length
            ).to(device)
    finally:
        tokenizer.truncation_side = old_trunc

    inputs_question = tokenizer(
        questions,
        return_tensors='pt',
        add_special_tokens=False,
        truncation=True,
        padding='max_length',
        max_length=max_question_length
        ).to(device)

    inputs_prompt = tokenizer(
        [prompt] * len(questions),
        return_tensors='pt',
        add_special_tokens=False,
        padding=False
        ).to(device)

    # get gist token id and number
    gist_token_id = getattr(model, 'gist_token_id', None)
    gist_token_num = 0

    if with_context:
        if compress and (gist_token_id is not None):
            # For ICAE-Flex: add gist tokens to the context ids if compress is True and gist token id is not None
            gist_token_num = max_context_length // compress_ratio
            gist_token_ids = torch.full(
                (inputs_context.input_ids.shape[0], gist_token_num), 
                gist_token_id, 
                dtype=torch.long, 
                device=device
            )
            # add gist tokens to the context ids
            inputs_context.input_ids = torch.cat([inputs_context.input_ids, gist_token_ids], dim=1)
            inputs_context.attention_mask = torch.cat([inputs_context.attention_mask, torch.ones_like(gist_token_ids)], dim=1)
        
        # with context
        input_ids = torch.cat([inputs_context.input_ids, inputs_question.input_ids, inputs_prompt.input_ids], dim=1)
        attention_mask = torch.cat([inputs_context.attention_mask, inputs_question.attention_mask, inputs_prompt.attention_mask], dim=1)
    else:
        # without context
        input_ids = torch.cat([inputs_question.input_ids, inputs_prompt.input_ids], dim=1)
        attention_mask = torch.cat([inputs_question.attention_mask, inputs_prompt.attention_mask], dim=1)

    if compress:
        # Compress
        ## Create context mask
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool).to(device)
        context_mask[:, :max_context_length+gist_token_num] = True
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ).tolist()
    else:
        # No compression
        ## Move paddings to the left for safety (the encoder-decoder model will do this as well during generation)
        input_ids, attention_mask, _ = move_padding_to(input_ids, attention_mask, padding_side="left")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            context_mask=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        ).tolist()
    
    # Extract the generated text
    outputs_text = []
    for sequence in outputs:
        if not compress:
            # If not compress, remove the repeated input tokens at the beginning
            valid_start_idx = input_ids.shape[1]
            sequence = sequence[valid_start_idx:]

        out_text = tokenizer.decode(sequence, skip_special_tokens=True).strip()

        if answer_extractor is not None:
            out_text = answer_extractor(out_text)
        outputs_text.append(out_text)

    return outputs_text


@torch.no_grad()
def beacon_predict(
    questions: list[str],
    contexts: list[str],
    model,
    tokenizer,
    device: str,
    prompt: str,
    answer_extractor: Optional[Callable] = None,
    max_new_tokens: int = 30,
    use_cache: bool = False,
    with_context: bool = True,
    max_question_length=128,
    max_context_length=512,
    **kwargs,
) -> list[str]:
    """
    Activation-beacon generation helper that avoids padding by generating one sample at a time.

    This matches the notebook-style usage:
      - no batch padding
      - `model._enable_beacon = True` (if present)
      - `use_cache=False` (recommended for beacon)

    Args:
        max_input_tokens: if set, truncates the prompt to this many tokens (prevents OOM on long contexts).
        max_question_length: the maximum length of the question tokens.
        max_context_length: the maximum length of the context tokens.

        Note: we do not restrict the maximum length of quesiton and context, but we derive the max input length from them.
         - max_input_tokens = max_question_length + max_context_length 
    """
    if len(questions) != len(contexts):
        raise ValueError(f"questions and contexts must be same length, got {len(questions)} and {len(contexts)}.")

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens([tokenizer.pad_token_id])[0]

    if hasattr(model, "_enable_beacon"):
        model._enable_beacon = True
    model.eval()

    max_input_tokens = max_question_length + max_context_length

    outputs_text: list[str] = []
    for q, ctx in zip(questions, contexts):
        if with_context:
            full_prompt = f"{ctx} {q} {prompt}"
        else:
            full_prompt = f"{q} {prompt}"

        tok_kwargs = {"return_tensors": "pt"}
        if max_input_tokens is not None:
            tok_kwargs.update({"truncation": True, "max_length": int(max_input_tokens)})

        old_trunc = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"   # keep the tail, matching SFT preprocessing
        try:
            inputs = tokenizer(full_prompt, **tok_kwargs).to(device)
        finally:
            tokenizer.truncation_side = old_trunc

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, prompt_len:]
        out_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        if answer_extractor is not None:
            out_text = answer_extractor(out_text)
        outputs_text.append(out_text)

    return outputs_text


@torch.no_grad()
def base_model_predict_single(
    questions: list[str],
    contexts: list[str],
    model,
    tokenizer,
    device: str,
    prompt: str,
    answer_extractor: Optional[Callable] = None,
    max_new_tokens: int = 30,
    max_question_length: int = 128,
    max_context_length: int = 512,
    with_context: bool = True,
    **kwargs,
) -> list[str]:
    """
    Predict single examples one by one, matching SFT base-model training input construction.

    This mirrors `SFTDataProcessor` + `DataCollatorForSFTBaseModel`:
      - context is truncated from the left and padded to `max_context_length`
      - prefix is tokenized as ONE string: (question_text + answer_prefix)
      - after concatenation, we pack all padding tokens to the RIGHT (so soft prompt virtual tokens stay left-aligned)
      - for generation stability, we trim trailing pads so the last token is never a pad token
    """
    if len(questions) != len(contexts):
        raise ValueError(f"questions and contexts must be same length, got {len(questions)} and {len(contexts)}.")

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    pad_id = tokenizer.pad_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Budget for the full prefix string "Question...{prompt}".
    max_prefix_length = int(max_question_length) + len(prompt_ids)

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    old_pad_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.truncation_side = "left"   # keep the tail, matching SFT preprocessing
    tokenizer.padding_side = "right"     # right-pad, then pack pads to the end

    outputs_text: list[str] = []
    try:
        for q, ctx in zip(questions, contexts):
            # IMPORTANT: do NOT add extra spaces here; formatting must be controlled by `prompt`/template.
            prefix_text = f"{q}{prompt}"

            if with_context:
                ctx_tok = tokenizer(
                    ctx,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding="max_length",
                    max_length=max_context_length,
                )
                prefix_tok = tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=max_prefix_length,
                )
                input_ids = torch.cat([ctx_tok.input_ids, prefix_tok.input_ids], dim=1).to(device)
                attention_mask = torch.cat([ctx_tok.attention_mask, prefix_tok.attention_mask], dim=1).to(device)
            else:
                prefix_tok = tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=max_prefix_length,
                ).to(device)
                input_ids = prefix_tok.input_ids
                attention_mask = prefix_tok.attention_mask

            # Safety: ensure pads are masked out
            attention_mask = attention_mask * (input_ids != pad_id).long()

            # Match training collator behavior: pack pads to the RIGHT across the full sequence.
            input_ids, attention_mask, _ = move_padding_to(
                input_ids, attention_mask, labels=None, padding_side="right"
            )

            # Generation stability: remove trailing pads so the last token is never a pad token.
            # (Otherwise `generate()` can start from a pad position and immediately emit EOS.)
            nonpad_len = int(attention_mask.sum(dim=1).item())
            if nonpad_len <= 0:
                nonpad_len = input_ids.shape[1]
            input_ids = input_ids[:, :nonpad_len]
            attention_mask = attention_mask[:, :nonpad_len]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = input_ids.shape[1]
            new_tokens = outputs[:, prompt_len:]
            out_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

            if answer_extractor is not None:
                out_text = answer_extractor(out_text)
            outputs_text.append(out_text)
    finally:
        tokenizer.truncation_side = old_trunc
        tokenizer.padding_side = old_pad_side

    return outputs_text


@torch.no_grad()
def base_model_predict(
    questions: list[str],
    contexts: list[str],
    model,
    tokenizer,
    device: str,
    prompt: str,
    answer_extractor: Optional[Callable] = None,
    max_new_tokens: int = 30,
    max_question_length: int = 128,
    max_context_length: int = 512,
    with_context: bool = True,
    **kwargs,
) -> list[str]:
    """
    Generation helper for standard (no-compression) causal LMs (including PEFT prompt-tuning adapters).

    We tokenize context/question/prompt separately to mirror SFT preprocessing budgets, then concatenate.
    """
    if len(questions) != len(contexts):
        raise ValueError(f"questions and contexts must be same length, got {len(questions)} and {len(contexts)}.")

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    # Match SFT preprocessing layout:
    #   input_ids = [context_ids (fixed max_context_length, right-padded)] + [prefix_ids (question + answer_prefix)]
    # Key details to match training:
    # - context truncation keeps the tail (truncate from left)
    # - prefix is tokenized as ONE string (avoids BPE boundary differences vs separate tokenization)
    # - generate per-sample to avoid batch padding making the "last token" a pad token
    pad_id = tokenizer.pad_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Budget for the full prefix string "Question...{prompt}".
    max_prefix_length = int(max_question_length) + len(prompt_ids)

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        outputs_text: list[str] = []
        for q, ctx in zip(questions, contexts):
            # Build the exact prefix used in SFT preprocessing: question_text + answer_prefix
            # Note: `questions` coming from Evaluator already include "Question: ...".
            prefix_text = f"{q}{prompt}"

            if with_context:
                ctx_tok = tokenizer(
                    ctx,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding="max_length",
                    max_length=max_context_length,
                )
                prefix_tok = tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=max_prefix_length,
                )
                input_ids = torch.cat([ctx_tok.input_ids, prefix_tok.input_ids], dim=1).to(device)
                attention_mask = torch.cat([ctx_tok.attention_mask, prefix_tok.attention_mask], dim=1).to(device)
            else:
                prefix_tok = tokenizer(
                    prefix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=max_prefix_length,
                ).to(device)
                input_ids = prefix_tok.input_ids
                attention_mask = prefix_tok.attention_mask

            # Safety: ensure pads are masked out
            if pad_id is not None:
                attention_mask = attention_mask * (input_ids != pad_id).long()

            # IMPORTANT for prompt tuning: move paddings to the right (so that the prefix virtual tokens are at the left)
            input_ids, attention_mask, _ = move_padding_to(
                input_ids, attention_mask, labels=None, padding_side="left"
            )

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = input_ids.shape[1]
            new_tokens = outputs[:, prompt_len:]
            out_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

            if answer_extractor is not None:
                out_text = answer_extractor(out_text)
            outputs_text.append(out_text)
    finally:
        tokenizer.truncation_side = old_trunc

    return outputs_text


@torch.no_grad()
def base_model_predict_batch(
    questions: list[str],
    contexts: list[str],
    model,
    tokenizer,
    device: str,
    prompt: str,
    answer_extractor: Optional[Callable] = None,
    max_new_tokens: int = 30,
    max_question_length: int = 128,
    max_context_length: int = 512,
    with_context: bool = True,
    **kwargs,
) -> list[str]:
    """
    Generation helper for standard (no-compression) causal LMs (including PEFT prompt-tuning adapters).

    We tokenize context/question/prompt separately to mirror SFT preprocessing budgets, then concatenate.
    """
    if len(questions) != len(contexts):
        raise ValueError(f"questions and contexts must be same length, got {len(questions)} and {len(contexts)}.")

    if len(questions) == 0:
        return []

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    # Match SFT preprocessing layout:
    #   input_ids = [context_ids (fixed max_context_length, right-padded)] + [prefix_ids (question + answer_prefix)]
    # Key details to match training:
    # - context truncation keeps the tail (truncate from left)
    # - prefix is tokenized as ONE string (avoids BPE boundary differences vs separate tokenization)
    pad_id = tokenizer.pad_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Budget for the full prefix string "Question...{prompt}".
    max_prefix_length = int(max_question_length) + len(prompt_ids)

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        # Build the exact prefix used in SFT preprocessing: question_text + answer_prefix
        # Note: `questions` coming from Evaluator already include "Question: ...".
        prefix_texts = [f"{q}{prompt}" for q in questions]

        if with_context:
            ctx_tok = tokenizer(
                contexts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                max_length=max_context_length,
            )
            prefix_tok = tokenizer(
                prefix_texts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding=True,
                max_length=max_prefix_length,
            )
            input_ids = torch.cat([ctx_tok.input_ids, prefix_tok.input_ids], dim=1).to(device)
            attention_mask = torch.cat([ctx_tok.attention_mask, prefix_tok.attention_mask], dim=1).to(device)
        else:
            prefix_tok = tokenizer(
                prefix_texts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding=True,
                max_length=max_prefix_length,
            ).to(device)
            input_ids = prefix_tok.input_ids
            attention_mask = prefix_tok.attention_mask

        # Safety: ensure pads are masked out
        if pad_id is not None:
            attention_mask = attention_mask * (input_ids != pad_id).long()

        # IMPORTANT for prompt tuning: move paddings to the right (so that the prefix virtual tokens are at the left)
        input_ids, attention_mask, _ = move_padding_to(
            input_ids, attention_mask, labels=None, padding_side="left"
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = input_ids.shape[1]
        new_tokens = outputs[:, prompt_len:]
        outputs_text = [tokenizer.decode(toks, skip_special_tokens=True).strip() for toks in new_tokens]

        if answer_extractor is not None:
            outputs_text = [answer_extractor(t) for t in outputs_text]
    finally:
        tokenizer.truncation_side = old_trunc

    return outputs_text




@torch.no_grad()
def compressing_score_candidates(
        questions: list[str],
        contexts: list[str],
        model: EncoderDecoderCompressorBase,
        tokenizer,
        device: str,
        prompt: str,
        candidate_token_ids: list[int],
        max_question_length: int = 128,
        max_context_length: int = 512,
        compress: bool = True,
        with_context: bool = True,
        compress_ratio: int = 4,
        **kwargs,
    ) -> list[int]:
    """
    Score candidate labels via a single forward pass through a compression model.

    Instead of generating text, this function compares logits at the last prompt
    position for a set of candidate token IDs (e.g. 'A', 'B', 'C') and returns
    the index of the highest-scoring candidate for each example.

    Args:
        questions: Formatted claim + options text (e.g. "Claim: ...\\nA) ...\\nB) ...").
        contexts: Context strings (can be empty when with_context=False).
        candidate_token_ids: Token IDs for the option letters (e.g. IDs of 'A', 'B', 'C').

    Returns:
        List of predicted candidate indices (one per example).
    """
    if compress and (not with_context):
        raise ValueError("compress=True but with_context=False.")

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"   # keep the tail, matching SFT preprocessing
    try:
        inputs_context = tokenizer(
            contexts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_context_length
        ).to(device)
    finally:
        tokenizer.truncation_side = old_trunc

    inputs_question = tokenizer(
        questions,
        return_tensors='pt',
        add_special_tokens=False,
        truncation=True,
        padding='max_length',
        max_length=max_question_length
    ).to(device)

    inputs_prompt = tokenizer(
        [prompt] * len(questions),
        return_tensors='pt',
        add_special_tokens=False,
        padding=False
    ).to(device)

    # Handle gist tokens for ICAE-Flex
    gist_token_id = getattr(model, 'gist_token_id', None)
    gist_token_num = 0

    if with_context:
        if compress and (gist_token_id is not None):
            gist_token_num = max_context_length // compress_ratio
            gist_token_ids = torch.full(
                (inputs_context.input_ids.shape[0], gist_token_num),
                gist_token_id,
                dtype=torch.long,
                device=device
            )
            inputs_context.input_ids = torch.cat([inputs_context.input_ids, gist_token_ids], dim=1)
            inputs_context.attention_mask = torch.cat([inputs_context.attention_mask, torch.ones_like(gist_token_ids)], dim=1)

        input_ids = torch.cat([inputs_context.input_ids, inputs_question.input_ids, inputs_prompt.input_ids], dim=1)
        attention_mask = torch.cat([inputs_context.attention_mask, inputs_question.attention_mask, inputs_prompt.attention_mask], dim=1)
    else:
        input_ids = torch.cat([inputs_question.input_ids, inputs_prompt.input_ids], dim=1)
        attention_mask = torch.cat([inputs_question.attention_mask, inputs_prompt.attention_mask], dim=1)

    if compress:
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool).to(device)
        context_mask[:, :max_context_length + gist_token_num] = True
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
        )
    else:
        input_ids, attention_mask, _ = move_padding_to(input_ids, attention_mask, padding_side="left")
        outputs = model.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # logits[:, -1, :] is correct because padding is moved to the left
    # (by _encode_and_prepare_lm_inputs for compress=True, or explicitly above for compress=False)
    last_logits = outputs.logits[:, -1, :]
    candidate_logits = last_logits[:, candidate_token_ids]
    predicted_indices = candidate_logits.argmax(dim=-1).tolist()
    return predicted_indices


@torch.no_grad()
def base_model_score_candidates(
        questions: list[str],
        contexts: list[str],
        model,
        tokenizer,
        device: str,
        prompt: str,
        candidate_token_ids: list[int],
        max_question_length: int = 128,
        max_context_length: int = 512,
        with_context: bool = True,
        **kwargs,
    ) -> list[int]:
    """
    Score candidate labels via a single forward pass through a standard causal LM.

    Args:
        questions: Formatted claim + options text.
        contexts: Context strings (can be empty when with_context=False).
        candidate_token_ids: Token IDs for the option letters.

    Returns:
        List of predicted candidate indices (one per example).
    """
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before inference.")

    pad_id = tokenizer.pad_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_prefix_length = int(max_question_length) + len(prompt_ids)

    old_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"

    try:
        prefix_texts = [f"{q}{prompt}" for q in questions]

        if with_context:
            ctx_tok = tokenizer(
                contexts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                max_length=max_context_length,
            )
            prefix_tok = tokenizer(
                prefix_texts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding=True,
                max_length=max_prefix_length,
            )
            input_ids = torch.cat([ctx_tok.input_ids, prefix_tok.input_ids], dim=1).to(device)
            attention_mask = torch.cat([ctx_tok.attention_mask, prefix_tok.attention_mask], dim=1).to(device)
        else:
            tok = tokenizer(
                prefix_texts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding=True,
                max_length=max_prefix_length,
            ).to(device)
            input_ids = tok.input_ids
            attention_mask = tok.attention_mask

        attention_mask = attention_mask * (input_ids != pad_id).long()
        input_ids, attention_mask, _ = move_padding_to(input_ids, attention_mask, padding_side="left")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        last_logits = outputs.logits[:, -1, :]
        candidate_logits = last_logits[:, candidate_token_ids]
        predicted_indices = candidate_logits.argmax(dim=-1).tolist()
    finally:
        tokenizer.truncation_side = old_trunc

    return predicted_indices


__all__ = [
    'compressing_predict_with_question_and_context',
    'beacon_predict',
    'base_model_predict',
    'base_model_predict_batch',
    'base_model_predict_single',
    'compressing_score_candidates',
    'base_model_score_candidates',
]
