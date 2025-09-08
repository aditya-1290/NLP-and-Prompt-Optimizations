from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained Phi-2-Instruct model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token for padding
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

# Define an instruction
instruction = "Translate the following English text to French: 'Hello, how are you?'"

# Encode the instruction
inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

# Generate response
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Extract the response after the instruction
response = generated_text[len(instruction):].strip()

print("Instruction:")
print(instruction)
print("\nGenerated Response:")
print(response)

# Simple evaluation
if "bonjour" in response.lower() or "comment" in response.lower():
    print("\nEvaluation: The response appears to be a correct translation.")
else:
    print("\nEvaluation: The response may not be accurate; it does not contain expected French words.")
