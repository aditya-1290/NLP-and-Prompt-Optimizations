from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the models to compare
models = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "microsoft/phi-2",
    "google/gemma-1.1b-it"
]

# Define a sample news article
article = "The stock market saw a significant rise today as tech companies reported strong earnings. Investors are optimistic about future growth in the sector."

# Define the prompt
prompt = f"Summarize the following news article: {article}"

# Function to generate response
def generate_response(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Generate and display responses
responses = {}
for model_name in models:
    print(f"Loading {model_name}...")
    response = generate_response(model_name, prompt)
    responses[model_name] = response
    print(f"Response from {model_name}: {response}")
    print("\n" + "-"*80 + "\n")

# Simple comparison
print("Comparison:")
print("All models generated summaries. Evaluate based on relevance, coherence, and quality.")
print("Note: Mistral may provide more detailed summaries, Phi-2 is concise, Gemma is balanced.")
