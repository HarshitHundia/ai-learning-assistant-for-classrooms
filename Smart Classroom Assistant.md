Smart Classroom Assistant: Interactive Demo & Code WalkthroughThis document provides a guided tour through the core functionalities and code implementation of the Smart Classroom Assistant. It's structured like a simplified Jupyter Notebook to help you understand the concepts alongside the code snippets.1. Project Setup and Core UtilitiesBefore diving into the main application, let's look at the essential imports and utility functions that power the assistant.1.1. Key ImportsWe leverage several powerful libraries for AI, web interaction, and file processing.import streamlit as st # For the web UI
import requests # For making HTTP requests to APIs
import json # For handling JSON data
import os # For environment variables and file paths

# For local model optimization
from optimum.intel.openvino import OVModelForSeq2SeqLM, OVModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoConfig

# For document processing
import pypdf # For PDF files
from docx import Document # For .docx files

# For voice utility (if used separately)
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyttsx3
import threading
import time
import asyncio
1.2. API Key ConfigurationSecurely managing API keys is crucial. For local development, environment variables are recommended. For the Canvas environment, keys are often injected.# Gemini API Key (replace with your actual key for local running)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Google Custom Search API Keys (for web search tool)
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "YOUR_GOOGLE_CSE_API_KEY_HERE")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "YOUR_GOOGLE_CSE_ID_HERE")

# In the full Streamlit app, these are handled via st.session_state
# and warnings are displayed if placeholders are still present.
2. Document Processing: Loading and Chunking NotesThe assistant can read various document types and prepare them for contextual question-answering by breaking them into manageable chunks.2.1. Reading Document ContentThis function handles .txt, .pdf, and .docx files.def read_document_content_from_file(file_path):
    """
    Reads the content of a text file, PDF, or DOCX and returns it as a string.
    Handles basic file reading errors and informs about OCR limitation for PDFs.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_extension == '.pdf':
            # This part would typically save to a temp file for pypdf
            # For brevity, imagine the file is directly accessible.
            try:
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:
                            content += page_text + "\n"
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return None
        elif file_extension == '.docx':
            # This part would typically save to a temp file for python-docx
            # For brevity, imagine the file is directly accessible.
            try:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
            except Exception as e:
                print(f"Error reading DOCX: {e}")
                return None
        else:
            print(f"Unsupported file type '{file_extension}'.")
            return None
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading file: {e}")
        return None

# Example Usage (in a full script, this would be part of file upload logic)
# notes_content = read_document_content_from_file("science_notes.txt")
# if notes_content:
#     print(f"Loaded {len(notes_content)} characters.")
2.2. Text Chunking and Relevant Chunk RetrievalLarge documents are split into smaller, overlapping chunks. When a question is asked, only the most relevant chunks are selected as context for the local model.def chunk_text(text, chunk_size=400, overlap=50):
    """Splits text into chunks of specified size with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_relevant_chunks(question, chunks, top_n=2):
    """
    Finds chunks most relevant to the question using simple keyword matching.
    Returns a concatenated string of the most relevant chunks.
    """
    question_keywords = set(word.lower() for word in question.split() if len(word) > 2)
    
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_keywords = set(word.lower() for word in chunk.split() if len(word) > 2)
        score = len(question_keywords.intersection(chunk_keywords))
        if score > 0:
            scored_chunks.append((score, i, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    relevant_chunks = [chunk for score, i, chunk in scored_chunks[:top_n]]

    return "\n---\n".join(relevant_chunks) if relevant_chunks else ""

# Example Usage
# sample_text = "Cells are the basic units of life. Prokaryotic cells lack a nucleus. Eukaryotic cells have a nucleus."
# chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
# print(f"Chunks: {chunks}")
# relevant = find_relevant_chunks("What are eukaryotic cells?", chunks)
# print(f"Relevant chunks: {relevant}")
3. Local AI Models: Optimized with OpenVINOFor fast, on-device contextual Q&A and emotion detection, we use OpenVINO to optimize Hugging Face models.3.1. Loading and Optimizing Flan-T5 for QAThe OVModelForSeq2SeqLM class handles the conversion and loading for sequence-to-sequence tasks like Q&A.# This function is cached by Streamlit to run only once
# @st.cache_resource
def load_local_qa_generator_ov():
    """Loads and optimizes the Flan-T5 model with OpenVINO."""
    model_id = "google/flan-t5-base"
    ov_model_dir = "./ov_flan_t5_base" # Directory to save/load optimized model
    os.makedirs(ov_model_dir, exist_ok=True)

    try:
        # from_transformers=True: load from Hugging Face
        # export=True: convert to OpenVINO IR
        # compile=True: compile for the current device (CPU by default)
        ov_model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, export=True, compile=True, save_directory=ov_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return ov_model, tokenizer
    except Exception as e:
        print(f"Error loading/optimizing Flan-T5 with OpenVINO: {e}")
        return None, None

# Global instances (loaded once at app startup)
# ov_qa_model, ov_qa_tokenizer = load_local_qa_generator_ov()

def get_local_answer(question_text, context_text=None):
    """
    Generates an answer using the OpenVINO-optimized Flan-T5 model.
    Includes a fallback trigger if the local model's answer is poor.
    """
    # Assume ov_qa_model and ov_qa_tokenizer are loaded globally
    # In the full Streamlit app, these are passed or accessed from global scope.
    if ov_qa_model is None or ov_qa_tokenizer is None:
        return "Local QA model not loaded."

    prompt_text = ""
    if context_text:
        prompt_text = f"Provide a comprehensive answer to the question based ONLY on the provided context. Context: {context_text}\n\nQuestion: {question_text}"
    else:
        prompt_text = f"Answer this question clearly and accurately: {question_text}"

    inputs = ov_qa_tokenizer(prompt_text, return_tensors="pt")
    outputs = ov_qa_model.generate(**inputs, max_new_tokens=128) # Limit output length
    answer = ov_qa_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Heuristic to detect poor answers from local model
    if not answer or len(answer) < 5 or any(phrase in answer.lower() for phrase in ["i cannot answer", "not in the response", "sorry", "could not find the answer"]):
        return "LOCAL_MODEL_FALLBACK_TRIGGERED"
    
    return answer
3.2. Loading and Optimizing Emotion ClassifierSimilar process for the OVModelForSequenceClassification model.# This function is cached by Streamlit to run only once
# @st.cache_resource
def load_emotion_classifier_ov():
    """Loads and optimizes the emotion classifier with OpenVINO."""
    model_id = "j-hartmann/emotion-english-distilroberta-base"
    ov_model_dir = "./ov_emotion_classifier"
    os.makedirs(ov_model_dir, exist_ok=True)

    try:
        ov_model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True, export=True, compile=True, save_directory=ov_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        return ov_model, tokenizer, config
    except Exception as e:
        print(f"Error loading/optimizing emotion classifier with OpenVINO: {e}")
        return None, None, None

# Global instances (loaded once at app startup)
# ov_emotion_model, ov_emotion_tokenizer, ov_emotion_config = load_emotion_classifier_ov()

def detect_emotion(text: str):
    """
    Detects the emotion in the given text using the OpenVINO-optimized classifier.
    """
    # Assume ov_emotion_model, ov_emotion_tokenizer, ov_emotion_config are loaded globally
    if ov_emotion_model is None or ov_emotion_tokenizer is None or ov_emotion_config is None:
        return {"label": "unknown", "score": 0.0, "message": "Emotion classifier not loaded."}
    
    try:
        inputs = ov_emotion_tokenizer(text, return_tensors="pt")
        outputs = ov_emotion_model(**inputs)
        
        # Assuming outputs.logits is a tensor, convert to numpy for argmax/softmax
        import numpy as np # Ensure numpy is imported
        probabilities = np.exp(outputs.logits.detach().numpy()) / np.sum(np.exp(outputs.logits.detach().numpy()), axis=1, keepdims=True)
        
        predicted_index = np.argmax(probabilities)
        label = ov_emotion_config.id2label[predicted_index]
        score = probabilities[0][predicted_index] # Access score from the first (and only) sample

        return {
            "label": label,
            "score": round(score, 2),
            "message": f"Detected Emotion: {label} (Score: {round(score, 2)})"
        }
    except Exception as e:
        print(f"Error detecting emotion with OpenVINO: {e}")
        return {"label": "error", "score": 0.0, "message": f"Error detecting emotion: {e}"}
4. Cloud AI Services: Gemini API with Tool UseFor general knowledge and complex generation tasks, we leverage the powerful Gemini API, which can also use external tools like Google Search.4.1. Real-time Search ToolThis function simulates (or integrates with) Google Custom Search.async def search_tool(query: str) -> str:
    """
    Performs a real-time web search using Google Custom Search JSON API.
    (Requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to be configured).
    """
    # Placeholder for actual API keys - ensure they are configured in your environment
    if GOOGLE_CSE_API_KEY == "YOUR_GOOGLE_CSE_API_KEY_HERE" or GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID_HERE":
        # Simulated responses for demo if keys are not set
        if "Koppal Lok Sabha" in query:
            return "Simulated: K. Rajashekar Basavaraj Hitnal (INC) won Koppal Lok Sabha in 2024."
        else:
            return f"Simulated: No specific real-time data for '{query}'. (API keys not configured)."

    search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()
        if search_results and 'items' in search_results:
            snippets = [f"Title: {item.get('title')}\nSnippet: {item.get('snippet')}\nLink: {item.get('link')}" for item in search_results['items'][:3]]
            return "\n---\n".join(snippets)
        else:
            return f"No relevant search results found for '{query}'."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to search service: {e}."
    except Exception as e:
        return f"An unexpected error occurred during search: {e}"
4.2. Getting Answers from Gemini (with Tool Integration)This function manages the conversation history and handles potential tool calls from Gemini.async def get_gemini_answer(question_text: str):
    """
    Fetches an answer from the Gemini 2.0 Flash API, incorporating tool use
    for real-time information and conversational memory.
    """
    # Assume chat_history is a global list managed by the Streamlit app
    # chat_history.append({"role": "user", "parts": [{"text": question_text}]}) # User input appended earlier

    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "search_tool",
                    "description": "Performs a real-time web search for current events, facts, or dynamic information.",
                    "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                }
            ]
        }
    ]

    payload = {"contents": chat_history, "tools": tools, "generationConfig": {"temperature": 0.9, "maxOutputTokens": 800}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("functionCall"):
            
            function_call = result["candidates"][0]["content"]["parts"][0]["functionCall"]
            function_name = function_call["name"]
            function_args = function_call["args"]

            print(f"Gemini called tool: {function_name} with {function_args}")
            chat_history.append({"role": "model", "parts": [{"functionCall": function_call}]})
            
            if function_name == "search_tool":
                tool_response_content = await search_tool(function_args.get("query"))
                chat_history.append({"role": "tool", "parts": [{"functionResponse": {"name": function_name, "response": {"text": tool_response_content}}}]})

                # Follow-up call to Gemini with tool response
                payload_with_tool_response = {"contents": chat_history, "tools": tools, "generationConfig": payload["generationConfig"]}
                response_after_tool = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload_with_tool_response))
                response_after_tool.raise_for_status()
                result_after_tool = response_after_tool.json()

                if result_after_tool.get("candidates") and result_after_tool["candidates"][0].get("content") and result_after_tool["candidates"][0]["content"].get("parts"):
                    final_answer = result_after_tool["candidates"][0]["content"]["parts"][0]["text"]
                    chat_history.append({"role": "model", "parts": [{"text": final_answer}]})
                    return final_answer
                else:
                    return "Sorry, couldn't get a clear answer after search."
            else:
                return "Sorry, unknown tool called."

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            direct_answer = result["candidates"][0]["content"]["parts"][0].get("text", "No text found.")
            chat_history.append({"role": "model", "parts": [{"text": direct_answer}]})
            return direct_answer
        else:
            return "Sorry, couldn't get a direct answer from Gemini."

    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}."
    except Exception as e:
        return f"An unexpected error occurred with Gemini: {e}"
5. Educational Features: Quiz Generation & SummarizationLeveraging Gemini's generative capabilities for creating study aids.5.1. Generating Quizzesasync def generate_quiz(topic: str, context: str):
    """Generates quiz questions based on a topic and provided context using Gemini."""
    quiz_prompt = f"""
    You are an educational assistant. Generate 5-7 multiple-choice quiz questions (with 4 options each) and their correct answers based on the following text about {topic}.
    Ensure the questions and options are clear and directly related to the text. Provide the correct answer letter.

    Text:
    {context}

    Format the output as follows:

    Question 1: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A/B/C/D]

    Question 2: ...
    """
    quiz_history_temp = [{"role": "user", "parts": [{"text": quiz_prompt}]}]
    payload = {"contents": quiz_history_temp, "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Could not generate quiz."
    except Exception as e:
        return f"Error generating quiz: {e}"
5.2. Summarizing Contentasync def summarize_content(text_to_summarize: str):
    """Summarizes provided text content using Gemini."""
    summary_prompt = f"""
    Please provide a concise and informative summary of the following text using a numbered list (1., 2., 3., etc.).
    Each numbered point should highlight a main point or key takeaway.

    Text:
    {text_to_summarize}
    """
    summary_history_temp = [{"role": "user", "parts": [{"text": summary_prompt}]}]
    payload = {"contents": summary_history_temp, "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "No summary generated."
    except Exception as e:
        return f"Error summarizing content: {e}"
6. Voice Assistant Utility (Console-based)While the Streamlit app is text-based, a separate console utility demonstrates voice interaction.6.1. Recording Audiodef record_audio(duration_seconds=5, samplerate=16000):
    """Records audio input from the microphone."""
    # ... (full implementation as in voice_assistant_code.py)
    # For brevity, actual recording logic is omitted here.
    print("🎤 Recording... (simulated)")
    # return "question.wav" # Simulated return
    pass
6.2. Transcribing Audio with Whisperdef transcribe_audio(audio_file_path):
    """Transcribes audio using the local Whisper model."""
    # ... (full implementation as in voice_assistant_code.py)
    # For brevity, actual transcription logic is omitted here.
    print(f"🎧 Transcribing {audio_file_path} with Whisper... (simulated)")
    # return "What is the capital of France?" # Simulated return
    pass
6.3. Speaking Answers with pyttsx3import threading
import time
import pyttsx3 # Ensure pyttsx3 is imported

global_engine = None
speaking_thread = None
stop_flag = threading.Event()

def _speak_in_thread(text_to_speak):
    """Helper function to run speech synthesis in a separate thread."""
    global global_engine
    if global_engine is None:
        global_engine = pyttsx3.init()
    stop_flag.clear()
    try:
        global_engine.say(text_to_speak)
        while global_engine.isBusy():
            if stop_flag.is_set():
                global_engine.stop()
                break
            time.sleep(0.1)
        global_engine.runAndWait()
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

def speak_answer(answer_text):
    """Starts speaking the given text in a new thread."""
    global speaking_thread
    if speaking_thread and speaking_thread.is_alive():
        stop_speaking()
        speaking_thread.join()
    speaking_thread = threading.Thread(target=_speak_in_thread, args=(answer_text,))
    speaking_thread.start()

def stop_speaking():
    """Signals the speaking thread to stop."""
    global global_engine, speaking_thread
    if speaking_thread and speaking_thread.is_alive():
        stop_flag.set()
        if global_engine and global_engine.isBusy():
            global_engine.stop()
7. Putting It All Together: The Streamlit App FlowThe smart_assistant_app_ui.py orchestrates all these components within a Streamlit web interface.# Simplified representation of the main Streamlit app logic
# (Full code is in smart_assistant_app_ui.py)

# Initialize session state variables
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# ... and other state variables

# File Uploader in sidebar
# uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["txt", "pdf", "docx"])
# if uploaded_file is not None:
#     # Logic to read file, chunk, and update session state
#     pass
# else:
#     # Logic to clear notes if file is removed
#     pass

# Chat display area
# for message in st.session_state.chat_history:
#     # Logic to display user/assistant/tool messages
#     pass

# User input
# user_input = st.chat_input("Ask a question or type a command...")
# if user_input:
#     st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_input}]})
    
#     # Command handling (e.g., "summarize notes", "quiz me")
#     # ...
    
#     # Q&A Routing Logic
#     if st.session_state.science_notes_content:
#         # Try local model first for contextual Q&A
#         relevant_context = find_relevant_chunks(user_input, st.session_state.science_notes_chunks)
#         if relevant_context and ov_qa_model:
#             local_answer = get_local_answer(user_input, context_text=relevant_context)
#             if local_answer == "LOCAL_MODEL_FALLBACK_TRIGGERED":
#                 # Fallback to Gemini if local model struggles
#                 asyncio.run(get_gemini_answer(user_input))
#             else:
#                 st.session_state.chat_history.append({"role": "model", "parts": [{"text": local_answer}]})
#         else:
#             # Fallback to Gemini if no relevant context or local model not available
#             asyncio.run(get_gemini_answer(user_input))
#     else:
#         # Always use Gemini if no document is loaded
#         asyncio.run(get_gemini_answer(user_input))
    
#     st.rerun() # Rerun to update UI
8. ConclusionThis interactive walkthrough demonstrates the modular design and key functionalities of the Smart Classroom Assistant. By combining local, optimized AI models with powerful cloud services and an intuitive Streamlit interface, the project delivers a versatile tool for enhancing educational experiences. The structured approach allows for easy understanding, maintenance, and future expansion.
