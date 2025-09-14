#!/usr/bin/env python3
"""
Darija LLM Fine-tuning Pipeline
Multi-stage fine-tuning for Moroccan Darija language models.
"""

import os
import yaml
import argparse
from time import time
from huggingface_hub import login
import wandb

from data_processor import DataProcessor
from model_manager import ModelManager
from trainer_manager import TrainerManager


class DarijaFineTuningPipeline:
    """Main pipeline for Darija LLM fine-tuning."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self._setup_environment()
        
        # Initialize managers
        self.data_processor = DataProcessor(self.config["environment"]["ds_cache_dir"])
        self.model_manager = ModelManager(self.config)
        self.trainer_manager = TrainerManager(self.config)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_environment(self):
        """Setup environment variables and authentication."""
        # Change working directory
        if self.config["environment"]["target_folder"]:
            print(f"Changing working directory to: {self.config['environment']['target_folder']}")
            os.chdir(self.config["environment"]["target_folder"])
        
        # Login to Hugging Face
        if self.config["environment"]["hf_token"]:
            print("Logging in to Hugging Face...")
            login(token=self.config["environment"]["hf_token"])
        
        # Login to Weights & Biases
        if self.config["environment"]["wandb_key"]:
            print("Logging in to Weights & Biases...")
            wandb.login(key=self.config["environment"]["wandb_key"])
        
        print("Environment setup complete!")
    
    def run_stage(self, stage: str):
        """Run a specific training stage."""
        print(f"\n{'='*50}")
        print(f"Running {stage.upper()} stage")
        print(f"{'='*50}")
        
        start_time = time()
        
        dataset = self._process_stage_data(stage)


        # Determine model path
        model_path = self.model_manager.determine_model_path(
            self.config["stages"], stage
        )
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer(model_path)
        
        # # Process data for this stage
        # dataset = self._process_stage_data(stage)
        
        # Configure model for training
        model, addcallbck = self._configure_model_for_stage(model, stage)
        
        # Create and run trainer
        trainer = self.trainer_manager.create_trainer(model, dataset, tokenizer, stage, addcallbck=addcallbck)
        self.trainer_manager.train_stage(trainer, stage)
        
        # Save model
        self.trainer_manager.save_model(
            trainer, stage, self.config["environment"]["hf_token"]
        )
        
        end_time = time()
        print(f"{stage.upper()} stage completed in {end_time - start_time:.2f} seconds")
    
    def _process_stage_data(self, stage: str):
        """Process data for a specific stage."""
        stage_config = self.config["stage_configs"][stage]
        datasets_config = stage_config["datasets"]
        
        if stage == "embed":
            return self.data_processor.process_embed_data(datasets_config)
        elif stage == "lora":
            if stage_config.get("phase") == "phase_1":
                return self.data_processor.process_lora_data_phase_1(datasets_config)
            elif stage_config.get("phase") == "phase_2":
                return self.data_processor.process_lora_data_phase_2(datasets_config)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _configure_model_for_stage(self, model, stage: str):
        """Configure model for specific training stage."""
        if stage == "embed":
            # Embedding-only training
            return self.model_manager.setup_embedding_training(model), False
        elif stage == "lora":
            # LoRA only training
            lora_config = self.config["stage_configs"]["lora"]["lora_config"]
            addcallbck = self.model_manager._verify_weight_sharing(model) and lora_config.get("train_embed", False)
            return self.model_manager.setup_lora_training(model, lora_config), addcallbck
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def run_pipeline(self):
        """Run the complete pipeline based on configured stages."""
        total_start_time = time()
        stages_to_run = self.config["stages"]
        
        print(f"Starting Darija Fine-tuning Pipeline")
        print(f"Stages to run: {stages_to_run}")
        
        for stage in stages_to_run:
            if stage not in ["embed", "lora"]:
                raise ValueError(f"Invalid stage: {stage}")
            
            self.run_stage(stage)
        
        total_end_time = time()
        print(f"\n{'='*50}")
        print(f"Pipeline completed successfully!")
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
        print(f"{'='*50}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Darija LLM Fine-tuning Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["embed", "lora"],
        help="Specific stages to run (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DarijaFineTuningPipeline(args.config)
    
    # Override stages if specified via command line
    if args.stages:
        pipeline.config["stages"] = args.stages
        print(f"Overriding config stages with: {args.stages}")
    
    # Run the pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()