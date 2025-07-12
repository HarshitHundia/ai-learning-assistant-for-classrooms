from transformers import pipeline

# Load a pretrained emotion classifier model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Sample student messages
inputs = [
    "What is this what kind of topics are these! Uff",
    "Wow, this was really fun to learn!",
    "I'm getting frustrated with this lesson.",
    "This makes sense now, thank you!",
]

# Detect and print emotions
for text in inputs:
    result = emotion_classifier(text)[0]
    print(f"\nText: {text}")
    print(f"Detected Emotion: {result['label']} (Score: {round(result['score'], 2)})")
    
