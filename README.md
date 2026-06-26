# LLMs-Training

This repository contains a modular, multi-stage pipeline for fine-tuning Large Language Models (LLMs), specifically optimized for Moroccan Darija language tasks. The pipeline leverages Hugging Face's `transformers`, `peft`, and `trl` libraries to manage training stages efficiently.

## Pipeline Overview

The framework is designed to run training in sequential stages (e.g., embedding optimization followed by LoRA fine-tuning). It handles:

* **Data Preparation:** Specialized processing for various datasets, including instructional and reasoning data.
* **Modular Training:** Support for different training types (Embeddings-only, LoRA).
* **Automatic Checkpoint Handling:** Automatically detects the previous stage's best checkpoint to continue training.
* **Integration:** Built-in support for Hugging Face Hub (saving/pushing) and Weights & Biases (tracking).

## Project Structure

```text
├── data_processor.py      # Handles dataset loading, cleaning, and formatting
├── main.py                # Entry point; manages the orchestration of stages
├── model_manager.py       # Handles model loading, PEFT/LoRA configuration, & weight freezing
├── trainer_manager.py     # Manages SFTTrainer, custom callbacks, and checkpointing
└── training_config.yaml   # Central configuration for models, datasets, and hyperparameters

```

## Configuration

All training parameters, environment variables (HF/WandB tokens), and dataset sources are controlled via `training_config.yaml`.

* **Stages:** Define the order of operations in the `stages` list.
* **Environment:** Set your `hf_token` and `wandb_key` here.
* **Stage-Specific Args:** Customize learning rates, batch sizes, and dataset mappings for each phase.

## Usage

### Prerequisites

Ensure you have the necessary dependencies installed (e.g., `torch`, `transformers`, `peft`, `trl`, `datasets`, `accelerate`).

### Running the Pipeline

You can run the full pipeline or override specific stages via the command line:

```bash
# Run with default config
python main.py --config training_config.yaml

# Override stages to run only LoRA
python main.py --config training_config.yaml --stages lora

```

## Key Features

* **Embedding Training:** Toggleable support for training embedding layers alongside the model.
* **Custom Callbacks:** Uses `SavePeftModelCallback` to ensure embedding layers are saved correctly during checkpointing when using PEFT.
* **Seamless Resumption:** The `TrainerManager` automatically checks for previous checkpoints in the `output_dir` to ensure training can resume without data loss.
* **Weight Sharing Awareness:** The `ModelManager` verifies weight sharing between the embedding and `lm_head` layers to prevent training conflicts.

---

*Developed for the purpose of efficient and scalable LLM fine-tuning.*

---
