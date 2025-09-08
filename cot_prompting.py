from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Define a math problem
problem = "If there are 3 apples and you buy 2 more apples, how many apples do you have in total?"

# Prompt without Chain-of-Thought (CoT)
prompt_without_cot = f"Question: {problem}\nAnswer:"

# Prompt with Chain-of-Thought (CoT)
prompt_with_cot = (
    f"Question: {problem}\n"
    "Let's think step by step.\n"
    "First, you have 3 apples.\n"
    "Then, you buy 2 more apples.\n"
    "So, total apples = 3 + 2 = 5.\n"
    "Answer:"
)

# Generate responses
response_without_cot = generate_text(prompt_without_cot)
response_with_cot = generate_text(prompt_with_cot)

print("Response without Chain-of-Thought prompting:")
print(response_without_cot)
print("\n" + "-"*50 + "\n")
print("Response with Chain-of-Thought prompting:")
print(response_with_cot)
