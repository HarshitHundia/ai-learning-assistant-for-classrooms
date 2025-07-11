from datasets import load_dataset

dataset = load_dataset("squad")  # or "emotion", "go_emotions", etc.
print("Loaded dataset with", len(dataset['train']), "examples.")
print("Example:", dataset['train'][0])