"""Training management utilities and custom callbacks."""

import os
import numpy as np
from typing import Dict, Any
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from torch.serialization import add_safe_globals
from accelerate import PartialState


class SavePeftModelCallback(TrainerCallback):
    """Custom callback to save PEFT models with embedding layers."""
    
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(
            checkpoint_folder, save_embedding_layers=True)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control


class TrainerManager:
    """Manages training configuration and execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = PartialState()
        self._setup_safe_globals()
    
    def _setup_safe_globals(self):
        """Setup safe globals for serialization."""
        add_safe_globals([
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.UInt32DType
        ])
    
    def create_training_config(self, stage: str) -> SFTConfig:
        """Create training configuration for a specific stage."""
        stage_config = self.config["stage_configs"][stage]
        common_args = self.config["common_training_args"]
        stage_training_args = stage_config["training_args"]
        
        # Merge common and stage-specific arguments
        training_args = {**common_args, **stage_training_args}
        training_args["output_dir"] = stage_config["output_dir"]
        training_args["max_seq_length"] = self.config["model"]["max_seq_length"]
        
        return SFTConfig(**training_args)
    
    def create_trainer(self, model, dataset, tokenizer, stage: str, addcallbck=False) -> SFTTrainer:
        """Create and configure trainer for a specific stage."""
        training_config = self.create_training_config(stage)
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=training_config,
        )
        
        # Add custom callback to save lara + embedding
        if addcallbck :
            trainer.add_callback(SavePeftModelCallback)
        
        return trainer
    
    def train_stage(self, trainer: SFTTrainer, stage: str):
        """Execute training for a stage with checkpoint resumption."""
        output_dir = self.config["stage_configs"][stage]["output_dir"]
        checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        
        print(f"Starting {stage} stage training...")
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            print("Starting training from scratch...")
            trainer.train()
    
    def save_model(self, trainer: SFTTrainer, stage: str, hf_token: str):
        """Save and upload trained model."""
        stage_config = self.config["stage_configs"][stage]
        model_name = stage_config["model_name"]
        
        self.state.wait_for_everyone()
        if self.state.is_main_process:
            print(f"Saving {stage} model...")
            
            # For instruct stage with LoRA, merge adapters first
            if stage == "lora":
                trainer.model.save_pretrained(f'{model_name}_lora')
                trainer.model.push_to_hub(repo_id=f'{model_name}_lora', token=hf_token)
                trainer.model = trainer.model.merge_and_unload()
            
            # Save locally
            trainer.model.save_pretrained(model_name)
            trainer.processing_class.save_pretrained(model_name)
            
            # Upload to hub
            trainer.model.push_to_hub(repo_id=model_name, token=hf_token)
            trainer.processing_class.push_to_hub(repo_id=model_name, token=hf_token)
            
            print(f"Model saved and uploaded.")
    
    def get_resume_checkpoint(self, path: str) -> str:
        """Get the last checkpoint from a directory."""
        return get_last_checkpoint(path) if os.path.isdir(path) else None