from datasets import load_dataset
from transformers import pipeline

# Load dataset
dataset = load_dataset("squad", split="train[:5]")  # test first 5 questions

# Load a text2text model (like Flan-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

# Run model on each question
for i, item in enumerate(dataset):
    question = item["question"]
    context = item["context"]
    expected = item["answers"]["text"][0]

    prompt = f"Question: {question} Context: {context}"

    model_answer = qa_model(prompt, max_length=50)[0]['generated_text']

    print(f"\n🔹 Q{i+1}: {question}")
    print(f"💬 Model Answer: {model_answer}")
    print(f"✅ Expected Answer: {expected}")