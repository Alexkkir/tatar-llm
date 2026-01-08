import os
import torch
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    PreTrainedTokenizerFast, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def train():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})
    
    dataset = load_dataset("text", data_files="data/raw_tatar.txt", split="train")
    
    dataset = dataset.filter(lambda x: len(x["text"]) > 20)
    
    block_size = 512
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping texts...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    
    config = GPT2Config(
        vocab_size=32000,
        n_positions=512,
        n_ctx=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model = GPT2LMHeadModel(config)
    print(f"Model Parameters: {model.num_parameters() / 1_000_000:.2f}M")
    
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained("./tatar_gpt_model")
    tokenizer.save_pretrained("./tatar_gpt_model")
    print("Model saved to ./tatar_gpt_model")

if __name__ == "__main__":
    train()
