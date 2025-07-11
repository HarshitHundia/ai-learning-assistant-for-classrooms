from datasets import load_dataset

# Load dataset (using a small slice for quick inspection)
# We'll load 'squad' again as it's familiar, but you can replace with 'natural_questions'
dataset = load_dataset("squad", split="train[:5]") # test first 5 questions

print("--- Dataset Information ---")

# 1. Print the dataset object itself
# This gives a summary of its structure, splits, and features.
print(f"\nDataset object: {dataset}")

# 2. Inspect the dataset's features (columns and their types)
# This shows you the names of the columns and the data type each column holds.
print(f"\nDataset Features (Columns):")
print(dataset.features)

# 3. Access individual items (rows) in the dataset
# A dataset behaves like a list, so you can access rows by index.
print(f"\n--- First Item (Row 0) ---")
first_item = dataset[0]
print(first_item)

# 4. Access specific fields within an item
# You can then drill down into the dictionary structure of an item.
print(f"\n--- Details of First Item ---")
print(f"Question: {first_item['question']}")
print(f"Context: {first_item['context']}")
# Note: 'answers' is a dictionary containing lists.
# Accessing the text of the first answer:
print(f"Expected Answer Text: {first_item['answers']['text'][0]}")
print(f"Expected Answer Start Index: {first_item['answers']['answer_start'][0]}")


# 5. Loop through a few items to see more examples (similar to your previous code)
print(f"\n--- Looping Through First 3 Items ---")
for i in range(3): # Look at the first 3 items
    item = dataset[i]
    question = item["question"]
    context = item["context"]
    expected_answer = item["answers"]["text"][0]

    print(f"\n--- Item {i+1} ---")
    print(f"Q: {question}")
    print(f"C: {context[:200]}...") # Print only first 200 chars of context for brevity
    print(f"A: {expected_answer}")