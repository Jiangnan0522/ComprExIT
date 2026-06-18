# Architecture: Gist Preprocessing Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Codebase                                │
│  <repo_root>/src/                           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  data_processing/preprocessing.py                            │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  class GistDataProcessor:                           │    │   │
│  │  │    - _tokenize_function()                           │    │   │
│  │  │    - _group_texts()                                 │    │   │
│  │  │    - _assign_task_metadata()                        │    │   │
│  │  │    - process_dataset()                              │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  Data Collators:                                    │    │   │
│  │  │    - DataCollatorForHierarchicalCompressor          │    │   │
│  │  │    - DataCollatorForICAE                            │    │   │
│  │  │    - DataCollatorForICAEFlex                        │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Import
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Activation Beacon Codebase                        │
│  <repo_root>/src/baselines/activation_beacon│
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  main/train_with_gist_preprocessing.py (NEW)                │   │
│  │                                                               │   │
│  │  1. Import GistDataProcessor                                 │   │
│  │  2. Initialize with tokenizer, context_length, etc.          │   │
│  │  3. Load raw datasets (JSON files)                           │   │
│  │  4. Process with GistDataProcessor                           │   │
│  │  5. Use GistCompatibleDataCollator                           │   │
│  │  6. Train with ActivationBeaconTrainer                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  src/args_with_gist.py (NEW)                                 │   │
│  │                                                               │   │
│  │  class GistTrainingArgs(BaseTrainingArgs):                   │   │
│  │    - ntp_ratio                                               │   │
│  │    - context_length                                          │   │
│  │    - generation_length                                       │   │
│  │    - streaming                                               │   │
│  │    - max_tokens                                              │   │
│  │    - ...                                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Existing Activation Beacon Components                       │   │
│  │                                                               │   │
│  │  - src/trainer.py (ActivationBeaconTrainer)                  │   │
│  │  - src/modeling_beacon.py                                    │   │
│  │  - src/llama/, src/mistral/, src/qwen2/                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────────┐
│  Raw JSON Files  │
│  {"text": "..."}│
└────────┬─────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 1: Load Dataset                                          │
│  ────────────────────────────────────────────────────────────  │
│  load_dataset('json', data_files=..., streaming=True)         │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 2: Initialize GistDataProcessor                          │
│  ────────────────────────────────────────────────────────────  │
│  GistDataProcessor(                                            │
│    tokenizer=tokenizer,                                        │
│    context_length=2048,                                        │
│    generation_length=2048,                                     │
│    ntp_ratio=1.0,                                              │
│    seed=42                                                     │
│  )                                                             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 3: Tokenize                                              │
│  ────────────────────────────────────────────────────────────  │
│  _tokenize_function(examples)                                  │
│    → tokenizer(examples["text"])                               │
│    → {"input_ids": [...], "attention_mask": [...]}            │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 4: Group Texts                                           │
│  ────────────────────────────────────────────────────────────  │
│  _group_texts(examples)                                        │
│    → Concatenate all input_ids                                 │
│    → Split into chunks of total_seq_length (4096)             │
│    → Drop small remainder                                      │
│                                                                 │
│  Before:                                                        │
│    Doc1: [100 tokens]                                          │
│    Doc2: [200 tokens]                                          │
│    Doc3: [3800 tokens]                                         │
│                                                                 │
│  After:                                                         │
│    Chunk1: [4096 tokens] (from Doc1 + Doc2 + part of Doc3)   │
│    Chunk2: [4096 tokens] (rest of Doc3 + ...)                │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 5: Assign Task Metadata                                  │
│  ────────────────────────────────────────────────────────────  │
│  _assign_task_metadata(datasets)                               │
│    → Add "for_ntp" field (True/False based on ntp_ratio)      │
│    → Add "reconstruction_segment" field (0, 1, 2, ... or -1)  │
│    → Expand reconstruction samples (1 → N segments)            │
│                                                                 │
│  Example with ntp_ratio=0.5:                                   │
│    Sample 1: for_ntp=True,  reconstruction_segment=-1         │
│    Sample 2: for_ntp=False, reconstruction_segment=0          │
│    Sample 2: for_ntp=False, reconstruction_segment=1          │
│    Sample 3: for_ntp=True,  reconstruction_segment=-1         │
│    Sample 4: for_ntp=False, reconstruction_segment=0          │
│    Sample 4: for_ntp=False, reconstruction_segment=1          │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 6: Limit Tokens/Samples                                  │
│  ────────────────────────────────────────────────────────────  │
│  if max_tokens:                                                │
│    max_samples = max_tokens * 1M / total_seq_length           │
│    dataset = dataset.take(max_samples)  # streaming           │
│                                                                 │
│  Example: max_tokens=3000 (3B tokens), total_seq_length=4096  │
│    → max_samples = 3,000,000,000 / 4096 ≈ 732,421 samples    │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 7: Shuffle (Training Set Only)                           │
│  ────────────────────────────────────────────────────────────  │
│  if shuffle_train_set and split == 'train':                    │
│    dataset = dataset.shuffle(seed=seed, buffer_size=10000)    │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 8: Create DataLoader                                     │
│  ────────────────────────────────────────────────────────────  │
│  DataLoader(                                                    │
│    dataset,                                                     │
│    batch_size=per_device_train_batch_size,                    │
│    collate_fn=GistCompatibleDataCollator(tokenizer)           │
│  )                                                             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 9: Collate Batch                                         │
│  ────────────────────────────────────────────────────────────  │
│  GistCompatibleDataCollator.__call__(batch)                    │
│    → Remove "for_ntp" field                                    │
│    → Remove "reconstruction_segment" field                     │
│    → Pad sequences to max length in batch                      │
│    → Convert to tensors                                        │
│                                                                 │
│  Input:                                                         │
│    [                                                            │
│      {"input_ids": [...], "for_ntp": True, ...},              │
│      {"input_ids": [...], "for_ntp": False, ...},             │
│    ]                                                            │
│                                                                 │
│  Output:                                                        │
│    {                                                            │
│      "input_ids": tensor([[...], [...]]),                      │
│      "attention_mask": tensor([[...], [...]]),                 │
│      "labels": tensor([[...], [...]])                          │
│    }                                                            │
└────────┬───────────────────────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────────┐
│  Step 10: Train                                                │
│  ────────────────────────────────────────────────────────────  │
│  ActivationBeaconTrainer.train()                               │
│    → Forward pass through model                                │
│    → Compute loss                                              │
│    → Backward pass                                             │
│    → Update weights                                            │
└────────────────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Script                               │
│         (train_with_gist_preprocessing.py)                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ↓               ↓               ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Model      │ │  Tokenizer   │ │  Arguments   │
│   (Beacon)   │ │              │ │  (Gist)      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │                └────────┬───────┘
       │                         │
       │                         ↓
       │              ┌─────────────────────┐
       │              │ GistDataProcessor   │
       │              │                     │
       │              │ - tokenize          │
       │              │ - group_texts       │
       │              │ - assign_metadata   │
       │              │ - limit_tokens      │
       │              └──────────┬──────────┘
       │                         │
       │                         ↓
       │              ┌─────────────────────┐
       │              │ Processed Dataset   │
       │              │                     │
       │              │ {input_ids,         │
       │              │  attention_mask,    │
       │              │  for_ntp,           │
       │              │  reconstruction_    │
       │              │  segment}           │
       │              └──────────┬──────────┘
       │                         │
       │                         ↓
       │              ┌─────────────────────┐
       │              │ GistCompatible      │
       │              │ DataCollator        │
       │              │                     │
       │              │ Remove Gist fields  │
       │              │ Pad sequences       │
       │              └──────────┬──────────┘
       │                         │
       │                         ↓
       │              ┌─────────────────────┐
       │              │ Batched Data        │
       │              │                     │
       │              │ {input_ids,         │
       │              │  attention_mask,    │
       │              │  labels}            │
       │              └──────────┬──────────┘
       │                         │
       └─────────────────────────┘
                    │
                    ↓
       ┌────────────────────────┐
       │ ActivationBeacon       │
       │ Trainer                │
       │                        │
       │ - compute_loss()       │
       │ - train()              │
       │ - evaluate()           │
       └────────────────────────┘
```

## File Structure

```
CompressIn/src/
│
├── data_processing/
│   ├── __init__.py
│   ├── preprocessing.py              ← Your GistDataProcessor
│   └── data_loading.py
│
└── baselines/activation_beacon/
    │
    ├── main/
    │   ├── train.py                  ← Original training script
    │   ├── train_with_gist_preprocessing.py  ← NEW: Gist-based training
    │   └── eval_*.py
    │
    ├── src/
    │   ├── __init__.py
    │   ├── args.py                   ← Original arguments
    │   ├── args_with_gist.py         ← NEW: Extended arguments
    │   ├── data.py                   ← Original data processing
    │   ├── trainer.py                ← ActivationBeaconTrainer
    │   ├── utils.py                  ← DefaultDataCollator
    │   └── ...
    │
    ├── configs/
    │   └── train_with_gist_example.json  ← NEW: Example config
    │
    ├── train_with_gist.sh            ← NEW: Training script
    ├── README_CN.md                  ← NEW: Chinese guide
    ├── QUICKSTART_GIST.md            ← NEW: Quick start
    ├── GIST_PREPROCESSING_INTEGRATION.md  ← NEW: Full guide
    ├── PREPROCESSING_COMPARISON.md   ← NEW: Comparison
    ├── INTEGRATION_SUMMARY.md        ← NEW: Summary
    └── ARCHITECTURE.md               ← NEW: This file
```

## Key Classes and Their Roles

### 1. GistDataProcessor (Your Code)
- **Location**: `data_processing/preprocessing.py`
- **Role**: Core preprocessing logic
- **Methods**:
  - `_tokenize_function()`: Tokenize raw text
  - `_group_texts()`: Concatenate and chunk
  - `_assign_task_metadata()`: Add NTP/reconstruction flags
  - `process_dataset()`: Main entry point

### 2. GistCompatibleDataCollator (New)
- **Location**: `main/train_with_gist_preprocessing.py`
- **Role**: Adapt Gist output for Activation Beacon
- **Functionality**:
  - Remove `for_ntp` field
  - Remove `reconstruction_segment` field
  - Pad sequences
  - Convert to tensors

### 3. GistTrainingArgs (New)
- **Location**: `src/args_with_gist.py`
- **Role**: Extended training arguments
- **New Fields**:
  - `ntp_ratio`
  - `context_length`
  - `generation_length`
  - `streaming`
  - `max_tokens`
  - etc.

### 4. ActivationBeaconTrainer (Existing)
- **Location**: `src/trainer.py`
- **Role**: Custom trainer for Activation Beacon
- **Functionality**:
  - Handle beacon-specific logic
  - Compute loss
  - Evaluation

## Execution Flow

```
User runs command
    ↓
Parse arguments (GistTrainingArgs)
    ↓
Load model and tokenizer
    ↓
Initialize GistDataProcessor
    ↓
Load raw datasets (JSON)
    ↓
Process with GistDataProcessor
    ├→ Tokenize
    ├→ Group texts
    ├→ Assign metadata
    └→ Limit tokens
    ↓
Create DataLoader with GistCompatibleDataCollator
    ↓
Initialize ActivationBeaconTrainer
    ↓
Training loop:
    ├→ Get batch from DataLoader
    ├→ Collate with GistCompatibleDataCollator
    ├→ Forward pass
    ├→ Compute loss
    ├→ Backward pass
    └→ Update weights
    ↓
Save checkpoint
```

## Memory Layout

### During Preprocessing (Streaming Mode)

```
Disk (JSON files)
    ↓ (read on-the-fly)
Buffer (10,000 samples)
    ↓ (process in batches)
Processed samples
    ↓ (yield one by one)
DataLoader
    ↓ (batch)
GPU
```

**Memory Usage**: O(buffer_size + batch_size) - Very efficient!

### During Preprocessing (Non-streaming Mode)

```
Disk (JSON files)
    ↓ (load all)
RAM (entire dataset)
    ↓ (process in parallel)
Processed dataset (in RAM)
    ↓ (sample)
DataLoader
    ↓ (batch)
GPU
```

**Memory Usage**: O(dataset_size) - May be high for large datasets

## Token Flow Example

```
Input: 3 documents

Doc1: "Hello world" (2 tokens)
Doc2: "How are you" (3 tokens)
Doc3: "I am fine" (3 tokens)

After tokenization:
[101, 102]  # Doc1
[103, 104, 105]  # Doc2
[106, 107, 108]  # Doc3

After grouping (total_seq_length=4):
Chunk1: [101, 102, 103, 104]  # Doc1 + part of Doc2
Chunk2: [105, 106, 107, 108]  # rest of Doc2 + Doc3

After assigning metadata (ntp_ratio=0.5):
Chunk1: {input_ids: [101, 102, 103, 104], for_ntp: True, reconstruction_segment: -1}
Chunk2: {input_ids: [105, 106, 107, 108], for_ntp: False, reconstruction_segment: 0}
Chunk2: {input_ids: [105, 106, 107, 108], for_ntp: False, reconstruction_segment: 1}

After collation:
Batch: {
  input_ids: tensor([[101, 102, 103, 104],
                     [105, 106, 107, 108],
                     [105, 106, 107, 108]]),
  attention_mask: tensor([[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]]),
  labels: tensor([[102, 103, 104, -100],
                  [106, 107, 108, -100],
                  [106, 107, 108, -100]])
}
```

## Comparison: Original vs Gist

### Original Preprocessing Flow
```
JSON → Tokenize → Truncate/Pad → Dataset → Collate → Model
         ↓           ↓
    [100 tokens] [4096 tokens]
                 (3996 padding!)
```

### Gist Preprocessing Flow
```
JSON → Tokenize → Group → Chunk → Metadata → Dataset → Collate → Model
         ↓          ↓       ↓
    [100 tokens] [4100]  [4096, 4]
                          (only 4 waste!)
```

## Summary

This architecture provides:

1. **Modularity**: Clear separation between preprocessing and training
2. **Reusability**: Direct reuse of your GistDataProcessor
3. **Compatibility**: Works with existing Activation Beacon code
4. **Efficiency**: Streaming support and minimal token waste
5. **Flexibility**: Easy to customize and extend

The integration is designed to be:
- **Non-invasive**: Doesn't modify your original code
- **Maintainable**: Clear structure and documentation
- **Extensible**: Easy to add new features
- **Testable**: Each component can be tested independently
