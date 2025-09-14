"""Model management utilities for loading and configuring models."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
from typing import Dict, Any, Optional, List
import os


class ModelManager:
    """Handles model loading, configuration, and parameter management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = PartialState()
        self.device_string = self.state.process_index
    
    def determine_model_path(self, stages: List[str], current_stage: str) -> str:
        """Determine which model to load based on the pipeline configuration."""
        stage_order = ["embed", "lora"]
        current_idx = stage_order.index(current_stage)
        
        # If this is the first stage being run, use base model
        if current_stage == stages[0]:
            return self.config["model"]["base_model_id"]
        
        # Otherwise, find the previous stage's checkpoint
        for i in range(current_idx - 1, -1, -1):
            prev_stage = stage_order[i]
            if prev_stage in stages:
                output_dir = self.config["stage_configs"][prev_stage]["output_dir"]
                checkpoint = self._get_last_checkpoint(output_dir)
                if checkpoint:
                    return checkpoint
        
        # Fallback to base model
        return self.config["model"]["base_model_id"]
    
    def _get_last_checkpoint(self, path: str) -> Optional[str]:
        """Get the last checkpoint from a directory."""
        return get_last_checkpoint(path) if os.path.isdir(path) else None
    
    def load_model_and_tokenizer(self, model_path: str):
        """Load model and tokenizer with proper configuration."""
        print(f"Loading model from: {model_path}")
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_new_tokens=1024,
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=model_config,
            torch_dtype=getattr(torch, self.config["model"]["torch_dtype"]),
            device_map={'': self.device_string},
            # device_map=None,
            attn_implementation=self.config["model"].get("attn_implementation", None)
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["tokenizer_hub"]
        )
        
        # Resize embeddings if needed
        if model.config.vocab_size != len(tokenizer):
            print("Resizing model token embeddings to match tokenizer vocab size.")
            model.resize_token_embeddings(len(tokenizer))
            print(f"Model vocab size after resizing: {model.config.vocab_size}")
            
            
            
        if tokenizer.pad_token is None:
            print("Pad token not set, using EOS token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding
        
        return model, tokenizer
    
    
    def setup_embedding_training(self, model):
        """Freeze all layers except embeddings and lm_head."""
        print("Setting up embedding-only training...")
        
        for name, param in model.named_parameters():
            if ('embed' in name) or ('lm_head' in name):
                param.requires_grad = True
                print(f'Trainable param: {name}')
            else:
                param.requires_grad = False
        
        self._print_trainable_parameters(model)
        self._verify_normalM_weight_sharing(model)
        
        return model

    def setup_lora_training(self, model, lora_config: Dict[str, Any]):
        """Setup LoRA training configuration."""
        print("Setting up LoRA training...")

        peft_config_kwargs = {
            "lora_alpha": lora_config["lora_alpha"],
            "lora_dropout": lora_config["lora_dropout"],
            "r": lora_config["r"],
            "bias": lora_config["bias"],
            "task_type": "CAUSAL_LM",
            "target_modules": lora_config["target_modules"]
        }
        
        weight_sharing = self._verify_weight_sharing(model)

        if lora_config.get("train_embed", False):
            if weight_sharing:
                print("Weight sharing verified between embeddings and lm_head.")
                print("Enabling training for embedding layers after LoRA application "
                    "(recommended if embedding and lm_head share weights or you will get a lot of problems if you use modules_to_save it will untie the sharing).")
            else:
                print("Weight sharing not verified, using modules_to_save=['lm_head', 'embed_tokens'].")
                peft_config_kwargs["modules_to_save"] = ["lm_head", "embed_tokens"]

        peft_config = LoraConfig(**peft_config_kwargs)
        model = get_peft_model(model, peft_config)

        if lora_config.get("train_embed", False) and weight_sharing:
            for name, param in model.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.requires_grad = True
                    print(f"Trainable param: {name}")

        self._print_trainable_parameters(model)
        return model

    def _print_trainable_parameters(self, model):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_pct = 100 * trainable_params / all_param
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_param:,} || "
              f"Trainable%: {trainable_pct:.2f}%")
    
    def _verify_weight_sharing(self, model):
        """Verify weight sharing between embeddings and lm_head."""
        lm_head = model.lm_head.weight
        embed_tokens = model.model.embed_tokens.weight
        return lm_head is embed_tokens