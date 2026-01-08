from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

def train_tokenizer(input_file, vocab_size=32000):
    print(f"Training tokenizer on {input_file} with vocab size {vocab_size}...")
    
    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train([input_file], trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    tokenizer.decoder = decoders.ByteLevel()
    
    os.makedirs("data", exist_ok=True)
    tokenizer.save("data/tokenizer.json")
    print("Tokenizer saved to data/tokenizer.json")

    sample_text = "Tatnet - Tatarça İnternet. Татар интернеты"
    encoded = tokenizer.encode(sample_text)
    print(f"Test encoding: '{sample_text}' -> {encoded.tokens}")

if __name__ == "__main__":
    train_tokenizer("data/raw_tatar.txt")

