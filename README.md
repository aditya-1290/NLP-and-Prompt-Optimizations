# NLP Tasks Scripts

This repository contains Python scripts demonstrating various NLP tasks using Hugging Face transformers library. Each script focuses on a specific prompting technique or model usage.

## Scripts Overview

### 1. zero_shot_classification.py
- **Purpose**: Performs zero-shot classification on text using the facebook/bart-large-mnli model.
- **Task**: Classifies sentences into positive, negative, or neutral categories without fine-tuning.
- **Usage**: Run `python zero_shot_classification.py`
- **Output**: Predicted categories and confidence scores for sample sentences.

### 2. few_shot_prompting.py
- **Purpose**: Demonstrates few-shot prompting for text generation using GPT-2.
- **Task**: Provides few examples of input-output pairs to generate a response to a new input.
- **Usage**: Run `python few_shot_prompting.py`
- **Output**: Generated translation response based on few-shot examples.

### 3. cot_prompting.py
- **Purpose**: Implements Chain-of-Thought (CoT) prompting for reasoning tasks using GPT-2.
- **Task**: Guides the model through step-by-step reasoning for a math problem.
- **Usage**: Run `python cot_prompting.py`
- **Output**: Responses with and without CoT prompting for comparison.

### 4. prompt_optimization.py
- **Purpose**: Experiments with prompt optimization for text generation using GPT-2.
- **Task**: Tests different prompt formats to improve response quality on the topic of the solar system.
- **Usage**: Run `python prompt_optimization.py`
- **Output**: Generated texts for different prompt variations.

### 5. response_control.py
- **Purpose**: Controls response style using temperature, max tokens, and top-p sampling with GPT-2.
- **Task**: Experiments with different parameters to adjust creativity and length.
- **Usage**: Run `python response_control.py`
- **Output**: Generated stories with varying parameters.

### 6. instruct_model.py
- **Purpose**: Uses Phi-2-Instruct model for instruction-following tasks.
- **Task**: Performs translation and evaluates the model's performance.
- **Usage**: Run `python instruct_model.py`
- **Output**: Generated translation response and simple evaluation.

### 7. compare_models.py
- **Purpose**: Compares performance of Mistral-Instruct, Phi-2-Instruct, and Gemma-1.1b-it models.
- **Task**: Generates summaries of a news article using each model.
- **Usage**: Run `python compare_models.py`
- **Output**: Summaries from each model for comparison.

## Dependencies

- Python 3.13+
- Hugging Face transformers: `pip install transformers`
- PyTorch: `pip install torch`
- (Optional) Accelerate for device mapping: `pip install accelerate`

## Installation

1. Clone or download the repository.
2. Install dependencies: `pip install -r requirements.txt` (if provided) or manually install as above.
3. Ensure you have sufficient RAM (at least 8GB recommended for larger models).

## Notes

- Some scripts load large models; they may take time to download and run.
- For models like Mistral and Gemma, ensure your system has adequate resources.
- Outputs are printed to the console for demonstration.

## License

This project is for educational purposes. Please check Hugging Face model licenses for usage.
