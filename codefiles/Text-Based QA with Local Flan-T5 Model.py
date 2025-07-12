# Import necessary libraries
from transformers import pipeline # For the local language model
import os # For file path operations

# Initialize the local Flan-T5 model for text generation (which can do Q&A)
# This model will be downloaded to your machine the first time you run this script.
# It's a "base" size model, offering a balance of performance and resource usage.
print("⏳ Loading local language model (google/flan-t5-base)... This may take a moment the first time.")
try:
    # Use the "text2text-generation" pipeline for Flan-T5.
    # It can generate answers based on implicit knowledge or provided context.
    local_qa_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    print("✅ Local language model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local model: {e}")
    print("Please ensure you have 'transformers' installed and an active internet connection for the initial download.")
    # Exit if the model can't be loaded, as the core functionality depends on it.
    exit()

def read_document_content(file_path):
    """
    Reads the content of a text file and returns it as a string.
    Handles basic file reading errors.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📄 Successfully loaded content from: {file_path}")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'. Please check the path.")
        return None
    except Exception as e:
        print(f"❌ Error reading file '{file_path}': {e}")
        return None

def get_local_answer(question_text, context_text=None):
    """
    Generates an answer to a question using a local Flan-T5 model,
    optionally using provided context.
    """
    print("🤔 Generating answer with local model (please wait)...")
    try:
        # Construct the prompt based on whether context is provided
        if context_text:
            # Modified prompt to encourage a more comprehensive answer
            prompt = f"Provide a comprehensive answer to the following question based ONLY on the provided context. Context: {context_text}\n\nQuestion: {question_text}"
        else:
            # If no context, it acts as a general Q&A (accuracy may vary).
            prompt = f"Answer this question clearly and accurately: {question_text}"

        # Generate the answer. max_length controls the length of the response.
        # Experiment with max_length if you need longer answers, but be mindful of verbosity.
        answer_output = local_qa_generator(prompt, max_length=150) # Increased max_length slightly as an experiment
        answer = answer_output[0]['generated_text'].strip()
        return answer
    except Exception as e:
        print(f"❌ Error during local model inference: {e}")
        return "Sorry, I encountered an error while trying to generate an answer locally."

# Main execution flow for text-based QA with contextual support
if __name__ == "__main__":
    print("\n--- Text-Based QA with Local Model (Contextual) ---")
    print("This assistant can answer questions based on a text file you provide.")
    print("Disclaimer: For general factual questions *without* context, local models like Flan-T5-base")
    print("may provide less accurate answers compared to cloud APIs like Gemini.")
    print("------------------------------------------------------\n")

    document_content = None
    while True:
        if document_content is None:
            file_path = input("📂 Enter the path to your text file (e.g., my_notes.txt), or type 'skip' for general Q&A: ").strip()
            if file_path.lower() == 'skip':
                print("Proceeding with general knowledge Q&A (no document context).")
            elif os.path.exists(file_path):
                document_content = read_document_content(file_path)
                if document_content is None: # If file reading failed, prompt again
                    continue
                print("\nReady to answer questions about your document!")
            else:
                print("❗ Invalid file path. Please try again.")
                continue

        question = input("📝 Enter your question (or 'new doc' to change file, 'exit' to quit): ").strip()

        if question.lower() == 'exit':
            print("Exiting module. Goodbye!")
            break
        elif question.lower() == 'new doc':
            document_content = None # Reset document content to prompt for new file
            print("\n--- Loading a new document ---")
            continue
        elif not question:
            print("Please enter a question.")
            continue

        # Get answer using the local model, passing the document content as context
        answer = get_local_answer(question, context_text=document_content)

        # Print the answer
        print("🤖 Assistant Answer:", answer)
        print("\n------------------------------------------------------\n")
