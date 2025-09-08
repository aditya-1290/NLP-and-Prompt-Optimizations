from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Define different prompts for optimization
prompts = [
    "Tell me about the solar system.",
    "Describe the solar system in detail, including the planets and the sun.",
    "The solar system consists of the sun and celestial bodies orbiting it. Provide a comprehensive overview of the solar system, covering its components, structure, and interesting facts."
]

# Generate and display responses for each prompt
for i, prompt in enumerate(prompts, 1):
    print(f"Prompt {i}: {prompt}")
    response = generate_text(prompt)
    print(f"Response {i}: {response}")
    print("\n" + "-"*80 + "\n")
