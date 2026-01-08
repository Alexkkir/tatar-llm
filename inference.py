import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
    return results

def main():
    print("Loading model...")
    model_path = "./tatar_gpt_model"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to CUDA")
    
    prompts = [
        "Татарстан - ул",
        "Казан шәһәре", 
        "Габдулла Тукай",
        "Бүген көн",
        "Мәктәптә укучылар",
        "Татар теле"
    ]
    
    print("\n--- GENERATION EXAMPLES ---\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated = generate_text(prompt, model, tokenizer)
        for i, text in enumerate(generated):
            print(f"Generated: {text}\n")
        print("-" * 30)

if __name__ == "__main__":
    main()
