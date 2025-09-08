from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Load Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map={"": "cpu"})

# System prompt for prompt tuning
SYSTEM_PROMPT = "You are a helpful and friendly chatbot. Respond accurately and sensibly to user queries."

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Prepend system prompt to user message
        full_prompt = f"{SYSTEM_PROMPT}\nUser: {request.message}\nAssistant:"

        # Encode the prompt
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(next(model.parameters()).device)

        # Generate response
        output_ids = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the response after the prompt
        response = generated_text[len(full_prompt):].strip()

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    with open("static/index.html", "r") as f:
        return f.read()
