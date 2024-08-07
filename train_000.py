import glob
import os
from base64 import b64decode
from io import BytesIO
from typing import Any, Optional, Union

import cv2
import datasets
import deepspeed
import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          Trainer, TrainingArguments)

from models.model_opt import GitOPTConfig, GitOPTForCausalLM

GitLLMForCausalLM = Any

# configファイルのパスを定義
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs/training_config_exp050_llama.yml')

# SupervisedDataset
class SupervisedDataset(Dataset):
    """Dataset for supervised learning"""

    def __init__(
        self,
        model_name: str,
        vision_model_name: str,
        loaded_dataset: datasets.GeneratorBasedBuilder,
        max_length: int = 128,
    ):
        super(SupervisedDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        self.processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True if "mpt" in model_name else False
        )
        if "llama" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        elif "mpt" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def __getitem__(self, index) -> dict:
        row = self.loaded_dataset[index]

        instruction = row["instruction"]
        question = row["inputs"]
        answer = row["outputs"]
        text = f"##Instruction: {instruction} ##Question: {question} ##Answer: {answer}"

        image_base64_str_list = row["image_base64_str"]
        img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        inputs = self.processor(
            text,
            img,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

def load_model(
    model_name: str, vision_model_name: str, num_image_with_embedding: Optional[int]
) -> GitLLMForCausalLM:
    if "opt" in model_name:
        git_config = GitOPTConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitOPTForCausalLM.from_pretrained(model_name, config=git_config)
    return model

def load_pretrained_weight(model: GitLLMForCausalLM, weight_path: str):
    weight = {}
    weight_path = glob.glob(f"{weight_path}/pytorch*.bin")
    for w in weight_path:
        weight_temp = torch.load(w, map_location="cpu")
        weight.update(weight_temp)
    model.load_state_dict(weight, strict=False)

def apply_lora_model(model: GitLLMForCausalLM, model_name: str, config: dict) -> GitLLMForCausalLM:
    peft_config = LoraConfig(**config["lora"])
    if "opt" in model_name:
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)
    elif "llama" in model_name:
        target_modules = []
        for m in peft_config.target_modules:
            target_modules += [
                f"model.layers.{i}.self_attn.{m}" for i in range(len(model.model.layers))
            ]

        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        model = model.base_model.model
    elif "mpt" in model_name:
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        model = model.base_model.model
    return model

def set_trainable_params(model: GitLLMForCausalLM, model_name: str, keys_finetune: list) -> None:
    if "mpt" in model_name:
        for name, p in model.transformer.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for name, p in model.model.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                p.requires_grad = True
            else:
                p.requires_grad = False

def get_dataset(config: dict) -> Union[Dataset, Dataset]:
    if config.get("dataset_type") is not None:
        # キャッシュを無視して再ダウンロード
        dataset_list = [
            datasets.load_dataset("MMInstruction/M3IT", i, cache_dir='/home/hata/.cache/huggingface/datasets', download_mode="force_redownload") for i in config["dataset_type"]
        ]
        train_dataset = ConcatDataset([d["train"] for d in dataset_list])
        val_dataset_list = []
        for d in dataset_list:
            try:
                val_dataset_list.append(d["validation"])
            except:
                print(f"{d['train']._info.config_name} has no validation set.")
        val_dataset = ConcatDataset(val_dataset_list)
    else:
        coco_datasets = datasets.load_dataset("MMInstruction/M3IT", "coco", cache_dir='/home/hata/.cache/huggingface/datasets', download_mode="force_redownload")
        train_dataset = coco_datasets["train"]
        val_dataset = coco_datasets["validation"]
    return train_dataset, val_dataset

def main(config_file: str = CONFIG_PATH):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    if os.getenv("WANDB_NAME") is not None:
        config["training"]["output_dir"] = os.path.join(
            config["training"]["output_dir"], os.getenv("WANDB_NAME")
        )

    deepspeed.init_distributed()

    model_name = config["settings"]["model_name"]
    vision_model_name = config["settings"]["vision_model_name"]
    num_image_with_embedding = config["settings"]["num_image_with_embedding"]

    train_dataset, val_dataset = get_dataset(config)

    max_length = config["settings"]["max_length"]
    keys_finetune = config["settings"]["keys_finetune"]

    training_args = TrainingArguments(**config["training"])

    model = load_model(model_name, vision_model_name, num_image_with_embedding)

    if config["use_lora"]:
        keys_finetune.append("lora")
        model = apply_lora_model(model, model_name, config)

    if config["settings"]["load_pretrained"] is not None:
        load_pretrained_weight(model, config["settings"]["load_pretrained"])
        print(
            f'Successfully loading pretrained weights from {config["settings"]["load_pretrained"]}'
        )

    set_trainable_params(model, model_name, keys_finetune)

    trainer = Trainer(
        model=model,
        train_dataset=SupervisedDataset(model_name, vision_model_name, train_dataset, max_length),
        eval_dataset=SupervisedDataset(model_name, vision_model_name, val_dataset, max_length),
        args=training_args,
    )

    with torch.cuda.amp.autocast():
        result = trainer.train()

    final_save_path = os.path.join(
        config["training"]["output_dir"], os.getenv("WANDB_NAME", "default") + "_final"
    )
    trainer.save_model(final_save_path)
    if "zero3" in config["training"]["deepspeed"]:
        trainer.model_wrapped.save_checkpoint(final_save_path)

if __name__ == "__main__":
    main()
