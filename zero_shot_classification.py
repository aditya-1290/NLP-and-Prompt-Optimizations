from transformers import pipeline

# Load the zero-shot classification pipeline with a pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate labels for classification
candidate_labels = ["positive", "negative", "neutral"]

# Sample sentences to classify
sentences = [
    "I love this product! It's amazing.",
    "This is terrible. I hate it.",
    "It's okay, nothing special.",
    "The service was excellent and fast.",
    "I'm disappointed with the quality."
]

# Perform zero-shot classification on each sentence
for sentence in sentences:
    result = classifier(sentence, candidate_labels)
    predicted_label = result['labels'][0]  # The top predicted label
    confidence = result['scores'][0]  # Confidence score for the top label
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Category: {predicted_label} (Confidence: {confidence:.2f})")
    print("-" * 50)
