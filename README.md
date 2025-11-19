# One Head to Rule Them All: Amplifying LVLM Safety through a Single Critical Attention Head(NeurIPS2025 Poster)

This repository contains the implementation of the paper "One Head to Rule Them All: Amplifying LVLM Safety
 through a Single Critical Attention Head" which proposes a novel approach to enhancing the safety of Large Vision-Language Models (LVLMs) by identifying and leveraging critical attention heads that are essential for safety.

## Overview

LVLMs have demonstrated impressive capabilities in multimodal understanding tasks, but they often exhibit degraded safety alignment compared to text-only LLMs. This project introduces a method to amplify LVLM safety by:

1. Investigating internal multi-head attention mechanisms
2. Identifying critical "safety" attention heads
3. Measuring deflection angle of hidden states to efficiently discriminate between safe and unsafe inputs
4. Implementing a defense mechanism that achieves near-perfect detection of harmful inputs while maintaining low false positive rates

## Requirements

- PyTorch
- Transformers
- LVLMs (LLaVA-v1.5-7B, Qwen2-VL-7B-Instruct, Aya-Vision-8B, Phi-3.5-Vision, etc.)

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The code works with several datasets:

- VLSafe: Contains harmful image-text pairs
- LLaVA-Instruct-80K: Used for safe dataset
- ShareGPT4V: Used for safe dataset testing
- JailbreakV-28K: Contains various jailbreak attack scenarios

Prepare your datasets in the following directory structure:

```
data/
├── JailBreakV_28K/
│   └── JailBreakV_28K.csv
├── ShareGPT4V/
│   └── sharegpt4v.csv
├── VLSafe/
│   └── vlsafe.csv
├── LLaVA-Instruct-80K/
│   └── safe.csv
```

## Using the Code

The repository is organized into several modules:

- `attack/`: Code for testing LVLMs with unsafe inputs
- `eval/`: Evaluation scripts using LLaMAGuard and Attack Success Rate calculation
- `detect/`: Implementation of the detection mechanism
- `defense/`: Implementation of the defense mechanism
- `head/`: Code for identifying and analyzing safety-critical attention heads
- `utils/`: Utility scripts for threshold determination and safe head selection

## Running Experiments(LLaVA-v1.5-7B)

### 1. Attack Testing

Test how LVLMs respond to unsafe inputs:

```bash
python attack/attack_vlm.py --model_path /path/to/models/llava-v1.5-7b \
                            --image_path /path/to/train2017 \
                            --csv_path data/VLSafe/vlsafe.csv
```

### 2. Safety Evaluation

Evaluate responses using LLaMAGuard:

```bash
python eval/eval.py --data_path ./results/attack/VLSafe/LLaVA/vlsafe.csv
```

Calculate Attack Success Rate:

```bash
python eval/asr.py
```

### 3. Safety Head Identification

Identify attention heads that are critical for safety:

```bash
python head/head_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/VLSafe/vlsafe.csv

python head/head_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/LLaVA-Instruct-80K/safe.csv
```

### 4. Safe Head Selection

Search for the optimal safety attention heads:

```bash
python utils/search_safe_head.py
```

### 5. Threshold Finding

Calculate deflection angles for different datasets:

```bash
# Calculate deflection angles for unsafe dataset
python detect/detect_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/VLSafe/vlsafe.csv \
       --hidden_layer -1
       --safe_heads [[8,2]] \

# Calculate deflection angles for safe dataset
python detect/detect_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/LLaVA-Instruct-80K/safe.csv \
       --hidden_layer -1
       --safe_heads [[8,2]] \
```

Determine the optimal threshold:

```bash
python utils/threshold.py \
       --file1 results/detect/llava-v1.5-7b/vlsafe_layer-1/defense_results.csv \
       --file2 results/detect/llava-v1.5-7b/safe_layer-1/defense_results.csv
```

### 6. Unsafe Input Detection

Detect potentially harmful inputs using identified safety heads:

```bash
python detect/detect_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/VLSafe/vlsafe.csv \
       --hidden_layer -1 \
       --safe_heads [[8,2]] \
       --threshold 2.16
```

### 7. Defense Implementation

Implement the defense mechanism to prevent harmful outputs:

```bash
python defense/defense_llava.py \
       --model_path /path/to/models/llava-v1.5-7b \
       --image_path /path/to/train2017 \
       --csv_path data/VLSafe/vlsafe.csv \
       --hidden_layer -1 \
       --safe_heads [[8,2]] \
       --threshold 2.16
```
