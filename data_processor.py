"""Data processing utilities for different training stages."""

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from typing import Dict, Any

# Notes the dataset returned must have only one column named text or messages
# text for raw text data it must not contain bos token at the beginning or eos token at the end, they will be added later automatically
# for messages it must be a list of dictionaries with keys "role" and "content"
class DataProcessor:
    """Handles data loading and preprocessing for different training stages."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
    
    def process_lora_data_phase_1(self, datasets_config: Dict[str, str]) -> Any:
        """Process bilingual datasets (Darija + English bilingual data)."""
        print("Loading lora datasets...")
        
        dar_dataset = load_dataset(
            datasets_config["darija"], 
            cache_dir=self.cache_dir
        )['train']
        print(f"Loaded Darija dataset: {len(dar_dataset)} examples")
        
        eng_dataset = load_dataset(
            datasets_config["english"], 
            cache_dir=self.cache_dir
        )['train'].shuffle(seed=42).select(range(800000))
        print(f"Loaded English dataset: {len(eng_dataset)} examples")
        
        instructen_dataset = load_dataset(
            datasets_config["instruct_en"], 
            cache_dir=self.cache_dir
        )['train']
        print(f"Loaded English instructions dataset: {len(instructen_dataset)} examples")
        
        def eninstruct_format(row):
            instruct = row["instruction"]+row["input"]
            resp = row["output"]

            doc_text = f"<|start_header_id|>user<|end_header_id|>\n\n{instruct}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{resp}"

            return {"text": doc_text}

        instructen_dataset = instructen_dataset.map(eninstruct_format)
        instructen_dataset = instructen_dataset.remove_columns([
                    col for col in instructen_dataset.column_names if col != "text"
                ])
        
        reasonen_dataset = load_dataset(
            datasets_config["reason_en"], 
            cache_dir=self.cache_dir
        )['train']
        print(f"Loaded Reasoning dataset: {len(reasonen_dataset)} examples")
        reasonen_dataset = reasonen_dataset.select(range(200000))
        def reason_format(row):
            messages = row["messages"]
            user_content = None
            assistant_content = None

            for i in range(len(messages) - 1):
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                    user_content = messages[i]["content"]
                    assistant_content = messages[i + 1]["content"]
                    break

            if user_content is None or assistant_content is None:
                return {"text": ""}

            doc_text = f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_content}"
            return {"text": doc_text} 
        
        reasonen_dataset = reasonen_dataset.map(reason_format)
        reasonen_dataset = reasonen_dataset.remove_columns("messages")
        
        dataset = concatenate_datasets([dar_dataset, eng_dataset, instructen_dataset, reasonen_dataset])
        
        
        # BOS_TOKEN = tokenizer.bos_token
        BOS_TOKEN = "<|begin_of_text|>"
        # EOS_TOKEN = "<|eot_id|>"
        def formatting_prompts_func(examples):
            # return { "text" : [BOS_TOKEN + example + EOS_TOKEN for example in examples["text"]] }
            return { "text" : [BOS_TOKEN + example for example in examples["text"]] }
        dataset = dataset.map(formatting_prompts_func, batched = True)
        # dataset = dataset.map(formatting_prompts_func, batched=True, load_from_cache_file=False)
        
        
        
        
        
        dataset = dataset.remove_columns([
            col for col in dataset.column_names if col != "text"
        ])
        dataset = dataset.shuffle(seed=42)
        print(f"Final lora dataset: {len(dataset)} examples")
        
        
        # dataset = dar_dataset.shuffle(seed=42).select(range(10000))
        # print(f"Final Monolingual dataset: {len(dataset)} examples")
        
        return dataset
    
    def process_embed_data(self, datasets_config: Dict[str, str]) -> Any:
        """Process embedding only datasets."""
        print("Loading embed datasets...")
        pass # placeholder for future implementation
        
    
    def process_lora_data_phase_2(self, datasets_config: Dict[str, str]) -> Any:
        """Process instruction-following datasets."""
        print("Loading embed datasets...")
        
        train_files = [
            hf_hub_download(
                repo_id=datasets_config["darija_instruct"],
                filename="data/train-00000-of-00002.parquet",
                repo_type="dataset"
            ),
            hf_hub_download(
                repo_id=datasets_config["darija_instruct"],
                filename="data/train-00001-of-00002.parquet",
                repo_type="dataset"
            )
        ]
        
        darija_instruct = load_dataset(
            "parquet", 
            data_files={"train": train_files}, 
            cache_dir=self.cache_dir
        )['train']
        
        darija_instruct = darija_instruct.remove_columns([
            col for col in darija_instruct.column_names if col != "messages"
        ])
        
        en_instruct = load_dataset(
            datasets_config["english_instruct"],
            cache_dir=self.cache_dir
        )['train']
        
        def to_chat_format(example):
            return {
                "messages": [
                    # {
                    #     "role": "system",
                    #     "content": example["instruction"].strip()
                    # },
                    {
                        "role": "user",
                        "content": example["instruction"].strip()+example["input"].strip()
                    },
                    {
                        "role": "assistant",
                        "content": example["output"].strip()
                    }
                ]
            }
        
        en_instruct = en_instruct.map(to_chat_format)
        en_instruct = en_instruct.remove_columns([
            col for col in en_instruct.column_names if col != "messages"
        ])
        
        
        ar_reason = load_dataset(
            datasets_config["ar_reason"],
            cache_dir=self.cache_dir
        )['train']       
        
        def to_messages(example):
            return {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["answer"]}
                ]
            }
        ar_reason = ar_reason.map(to_messages)
        ar_reason = ar_reason.remove_columns([
            col for col in ar_reason.column_names if col != "messages"
        ])


        reasonen_dataset = load_dataset(
            datasets_config["reason_en"], 
            cache_dir=self.cache_dir
        )['train']
        # print(f"Loaded Reasoning dataset: {len(reasonen_dataset)} examples")
        reasonen_dataset = reasonen_dataset.select(range(10000))
        reasonen_dataset = reasonen_dataset.remove_columns([
            col for col in reasonen_dataset.column_names if col != "messages"
        ])
        
        tulu_cross = load_dataset(
            datasets_config["tulu_cross_answer"], 
            cache_dir=self.cache_dir
        )['train']
        
        tulu_cross = tulu_cross.remove_columns([
            col for col in tulu_cross.column_names if col != "messages"
        ])
        
        tulu_cross = tulu_cross.shuffle(seed=42).select(range(5000))

        
        
        tulu_transl = load_dataset(
            datasets_config["tulu_translations"], 
            cache_dir=self.cache_dir
        )['train']
        
        tulu_transl = tulu_transl.remove_columns([
            col for col in tulu_transl.column_names if col != "messages"
        ]).select(range(200000)) 

        dataset = concatenate_datasets([darija_instruct, en_instruct, ar_reason, reasonen_dataset, tulu_cross, tulu_transl])
        
        def transform(example):
            return {
                "prompt": [msg for msg in example["messages"] if msg["role"] == "user"],
                "completion": [msg for msg in example["messages"] if msg["role"] == "assistant"]
            }
        dataset = dataset.map(transform)
        dataset = dataset.remove_columns("messages")
        
        def has_completion(example):
            return bool(example["completion"]) and len(example["completion"]) > 0
        dataset= dataset.filter(has_completion)


        dataset = dataset.shuffle(seed=42)
        
        print(f"Processed embed dataset: {len(dataset)} examples")
        return dataset