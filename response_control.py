from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, temperature=1.0, max_length=100, top_p=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Define a prompt
prompt = "Write a short story about a robot."

# Experiment with temperature
print("Experimenting with Temperature:")
temperatures = [0.2, 0.7, 1.0]
for temp in temperatures:
    response = generate_text(prompt, temperature=temp)
    print(f"Temperature {temp}: {response}")
    print("\n" + "-"*80 + "\n")

# Experiment with max tokens
print("Experimenting with Max Tokens:")
max_lengths = [50, 100, 150]
for max_len in max_lengths:
    response = generate_text(prompt, max_length=max_len)
    print(f"Max Tokens {max_len}: {response}")
    print("\n" + "-"*80 + "\n")

# Experiment with top-p sampling
print("Experimenting with Top-P Sampling:")
top_ps = [0.5, 0.9, 1.0]
for top_p in top_ps:
    response = generate_text(prompt, top_p=top_p)
    print(f"Top-P {top_p}: {response}")
    print("\n" + "-"*80 + "\n")
