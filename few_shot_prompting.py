from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Few-shot examples: input-output pairs
few_shot_examples = [
    ("Translate English to French:\nEnglish: Hello, how are you?\nFrench:", " Bonjour, comment ça va?"),
    ("Translate English to French:\nEnglish: What is your name?\nFrench:", " Quel est ton nom?"),
    ("Translate English to French:\nEnglish: I love programming.\nFrench:", " J'aime programmer.")
]

# Construct the prompt with few-shot examples
prompt = ""
for inp, out in few_shot_examples:
    prompt += inp + out + "\n"

# New input for generation
new_input = "Translate English to French:\nEnglish: Where is the library?\nFrench:"

prompt += new_input

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text with GPT-2
output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Extract the generated response after the new input prompt
generated_response = generated_text[len(prompt):].strip()

print("Prompt:")
print(prompt)
print("\nGenerated Response:")
print(generated_response)
