import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import json
import asyncio
import os
from transformers import pipeline # For the local Flan-T5 model and emotion classifier
import string # Import string module for punctuation
import pypdf # For reading PDF files
from docx import Document # For reading .docx files
import sys # Import sys to read stdin

# --- Configuration & Global Variables ---
# IMPORTANT: For Canvas environment, __api_key__ is provided as a global variable.
# When running locally outside Canvas, you MUST provide your own Gemini API key.
# It's recommended to set it as an environment variable (e.g., export GEMINI_API_KEY="your_key")
# or hardcode it for quick testing.

# Attempt to get the API key provided by the Canvas environment or from environment variables
GEMINI_API_KEY = globals().get('__api_key__', os.getenv("GEMINI_API_KEY", ""))

# If running locally and GEMINI_API_KEY env var is not set,
# you can uncomment the line below and replace with your actual key for testing:
GEMINI_API_KEY = "AIzaSyAgrpayn0eNJy-hqm8OrOOGf6mFrzyaGPE" # <--- YOUR GEMINI API KEY HAS BEEN ADDED HERE!

# Debug print to verify API key is being picked up
if GEMINI_API_KEY:
    print(f"DEBUG: Using Gemini API Key (first 5 chars): {GEMINI_API_KEY[:5]}...")
else:
    print("DEBUG: Gemini API Key is empty. Gemini API calls will likely fail with 403 Forbidden.")

# --- Real-time Search API Configuration ---
GOOGLE_CSE_API_KEY = "AIzaSyAgrpayn0eNJy-hqm8OrOOGf6mFrzyaGPE" # <--- UPDATED WITH YOUR API KEY
GOOGLE_CSE_ID = "e16d416d56a8a40b2"       # <--- UPDATED WITH YOUR SEARCH ENGINE ID (CX)

# Local model setup for contextual QA. This will be downloaded once.
print("⏳ Loading local language model (google/flan-t5-base)... This may take a moment the first time.")
try:
    local_qa_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    print("✅ Local language model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local model: {e}")
    print("Please ensure you have 'transformers' installed (`pip install transformers`) and an active internet connection for the initial download.")
    # In a production system, you might handle this more gracefully, but for now, we exit.
    sys.exit(1) # Exit if critical model fails to load

# --- Emotion Classifier Setup ---
print("⏳ Loading emotion classifier model (j-hartmann/emotion-english-distilroberta-base)...")
try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    print("✅ Emotion classifier loaded successfully.")
except Exception as e:
    print(f"❌ Error loading emotion classifier: {e}")
    print("Please ensure you have 'transformers' installed and an active internet connection for the initial download.")
    emotion_classifier = None # Set to None if loading fails, so detect_emotion handles it.

# Global variables for persistent state across UI calls
chat_history = []
science_notes_content = None
loaded_file_name = None
science_notes_chunks = []
last_generated_quiz = None
last_generated_summary = None

# --- External Tool: Google Search ---
async def search_tool(query: str) -> str:
    """
    Performs a real-time web search for the given query using Google Custom Search JSON API.
    Requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to be configured.
    """
    if GOOGLE_CSE_API_KEY == "YOUR_GOOGLE_CSE_API_KEY" or GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID":
        print("❗ Warning: Google Custom Search API keys are not configured. Using simulated fallback.")
        if "Koppal Lok Sabha" in query or "MP Koppal" in query or "Member of Parliament Koppal" in query:
            return """
            (Simulated) Latest news (June 2024): K. Rajashekar Basavaraj Hitnal (Indian National Congress) won the Koppal Lok Sabha constituency in the 2024 General Elections. He defeated Basavaraj Kyavator (Bharatiya Janata Party).
            Source: Election Commission of India results, leading news outlets.
            """
        elif "Gangavati Constituency MLA" in query or "MLA of Gangavati" in query:
            return """
            (Simulated) Latest news (May 2023 Karnataka Assembly Elections): G. Janardhana Reddy (Kalyana Rajya Pragathi Paksha) is the current MLA for Gangavati Assembly Constituency.
            Source: Election Commission of India results, state assembly websites.
            """
        elif "today's weather" in query or "weather today" in query or "weather in Bengaluru" in query:
            return "(Simulated) The weather today in Bengaluru is partly cloudy with a high of 30°C and a low of 22°C. There's a 20% chance of rain."
        elif "prime minister of india" in query.lower():
            return "(Simulated) The current Prime Minister of India is Narendra Modi, who assumed office on May 26, 2014."
        elif "current president of india" in query.lower() or "who is the president of india" in query.lower():
            return "(Simulated) The current President of India is Droupadi Murmu. She assumed office on 25 July 2022."
        else:
            return f"(Simulated) No specific real-time data found for '{query}'. This is a simulated search result because API keys are not configured."

    search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"

    print(f"🔍 Performing real web search for: '{query}' using Google Custom Search API...")
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()

        if search_results and 'items' in search_results:
            snippets = []
            for item in search_results['items'][:3]:
                title = item.get('title', 'No Title')
                snippet = item.get('snippet', 'No Snippet')
                link = item.get('link', '#')
                snippets.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}")
            return "\n---\n".join(snippets)
        else:
            return f"No relevant search results found for '{query}' from real API."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during real search API call: {e}")
        return f"Error connecting to real search service: {e}. Check your API keys and network."
    except Exception as e:
        print(f"❌ An unexpected error occurred during real search: {e}")
        return f"An unexpected error occurred during search: {e}"

# --- RECORD VOICE INPUT (Simulated for UI integration) ---
def record_audio_simulated():
    """Placeholder for UI-driven audio recording."""
    print("🎤 Recording (handled by UI)...")
    return "simulated_audio.wav"

# --- TRANSCRIBE USING LOCAL WHISPER (Simulated for UI integration) ---
def transcribe_audio_simulated(audio_file_path):
    """Placeholder for UI-driven transcription."""
    print("🎧 Transcribing (handled by UI)...")
    return "Simulated transcription of voice input."

# --- Custom Document Content Loading ---
def read_document_content_from_file(file_path):
    """
    Reads the content of a text file, PDF, or DOCX and returns it as a string.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        content = ""

        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_extension == '.pdf':
            try:
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:
                            content += page_text + "\n"
                if not content.strip():
                    print("❗ Warning: PDF seemed to contain no extractable text. This often happens with scanned PDFs (images of text). OCR is not supported in this version.")
                    return None
            except pypdf.errors.PdfReadError:
                print(f"❌ Error reading PDF '{file_path}': It might be corrupted or encrypted.")
                return None
            except Exception as pdf_e:
                print(f"❌ An unexpected error occurred while reading PDF '{file_path}': {pdf_e}")
                return None
        elif file_extension == '.docx':
            try:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
            except Exception as docx_e:
                print(f"❌ Error reading DOCX '{file_path}': {docx_e}. Ensure it's a valid .docx file.")
                return None
        else:
            print(f"❌ Error: Unsupported file type '{file_extension}'. Only .txt, .pdf, and .docx are supported.")
            return None

        print(f"📄 Successfully loaded content from: {file_path}")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'. Please check the path.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading file '{file_path}': {e}")
        return None

# --- Local Flan-T5 Answer Generation (Contextual QA) ---
def get_local_answer(question_text, context_text=None):
    """
    Generates an answer to a question using a local Flan-T5 model,
    optionally using provided context.
    """
    print("🤔 Generating answer with local model (please wait)...")
    try:
        if local_qa_generator is None:
            return "Local QA model is not loaded. Cannot generate answer."

        if context_text:
            prompt = f"Provide a comprehensive answer to the question based ONLY on the provided context. Context: {context_text}\n\nQuestion: {question_text}"
        else:
            prompt = f"Answer this question clearly and accurately: {question_text}"

        answer_output = local_qa_generator(prompt, max_length=256) 
        answer = answer_output[0]['generated_text'].strip()
        return answer
    except Exception as e:
        print(f"❌ Error during local model inference: {e}")
        return "Sorry, I encountered an error while trying to generate an answer locally."

# --- Emotion Detection Function ---
def detect_emotion(text: str):
    """
    Detects the emotion in the given text using the pre-loaded emotion classifier.
    Returns the detected label and its score.
    """
    if emotion_classifier is None:
        return {"label": "unknown", "score": 0.0, "message": "Emotion classifier not loaded."}
    try:
        result = emotion_classifier(text)[0]
        return {
            "label": result['label'],
            "score": round(result['score'], 2),
            "message": f"Detected Emotion: {result['label']} (Score: {round(result['score'], 2)})"
        }
    except Exception as e:
        return {"label": "error", "score": 0.0, "message": f"Error detecting emotion: {e}"}

# --- GEMINI API Answer Generation (General QA, Tool Use, Conversational Memory) ---
async def get_gemini_answer(question_text: str):
    """
    Fetches an answer from the Gemini 2.0 Flash API, incorporating tool use
    for real-time information and conversational memory.
    """
    global chat_history

    chat_history.append({"role": "user", "parts": [{"text": question_text}]})

    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "search_tool",
                    "description": "Performs a real-time web search for current events, facts, or dynamic information that is not available in the model's training data or provided context.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The search query to execute to get up-to-date information."
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    ]

    payload = {
        "contents": chat_history,
        "tools": tools,
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 800
        }
    }

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

            print(f"🤖 Gemini decided to call a tool: {function_name} with args {function_args}")

            chat_history.append({"role": "model", "parts": [{"functionCall": function_call}]})
            
            if function_name == "search_tool":
                tool_response_content = await search_tool(function_args.get("query"))
                
                chat_history.append({
                    "role": "tool",
                    "parts": [{"functionResponse": {"name": function_name, "response": {"text": tool_response_content}}}]
                })

                payload_with_tool_response = {
                    "contents": chat_history,
                    "tools": tools,
                    "generationConfig": payload["generationConfig"]
                }
                response_after_tool = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload_with_tool_response))
                response_after_tool.raise_for_status()
                result_after_tool = response_after_tool.json()

                if result_after_tool.get("candidates") and result_after_tool["candidates"][0].get("content") and \
                   result_after_tool["candidates"][0]["content"].get("parts"):
                    final_answer = result_after_tool["candidates"][0]["content"]["parts"][0]["text"]
                    chat_history.append({"role": "model", "parts": [{"text": final_answer}]})
                    return final_answer
                else:
                    print("❗ Warning: Gemini API (after tool) response structure unexpected or content missing.")
                    print("Full response (after tool):", result_after_tool)
                    return "Sorry, I processed the search but couldn't get a clear answer from Gemini."
            else:
                return "Sorry, Gemini tried to call an unknown tool."

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            direct_answer = result["candidates"][0]["content"].get("parts")[0].get("text", "No text found in response.")
            chat_history.append({"role": "model", "parts": [{"text": direct_answer}]})
            return direct_answer
        else:
            print("❗ Warning: Gemini API response structure unexpected or content missing.")
            print("Full response:", result)
            return "Sorry, I couldn't get a direct answer from Gemini."

    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Gemini API: {e}. Check your internet connection or API key.")
        return "Sorry, I'm having trouble connecting to the answer service."
    except Exception as e:
        print(f"❌ An unexpected error occurred with Gemini: {e}")
        return "An unexpected error occurred while getting the answer from Gemini."

# --- New Feature: Quiz Generation ---
async def generate_quiz(topic: str, context: str):
    """Generates quiz questions based on a topic and provided context using Gemini."""
    global last_generated_quiz

    print(f"📝 Generating a quiz on '{topic}' from the provided notes...")
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
    
    payload = {
        "contents": quiz_history_temp,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 800
        }
    }
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            quiz_text = result["candidates"][0]["content"].get("parts")[0].get("text", "No quiz generated.")
            last_generated_quiz = quiz_text
            return quiz_text
        else:
            last_generated_quiz = None
            return "Could not generate quiz. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        last_generated_quiz = None
        return f"Error generating quiz: {e}"
    except Exception as e:
        last_generated_quiz = None
        return f"An unexpected error occurred during quiz generation: {e}"

# --- New Feature: Identify Important Questions from Quiz ---
async def identify_important_questions_from_quiz(quiz_text: str):
    """Identifies and extracts important questions from a given quiz text using Gemini."""
    print("🧠 Identifying important questions from the quiz...")
    importance_prompt = f"""
    From the following quiz questions, identify the 2 to 3 most important or foundational questions.
    Present them as a numbered list.
    
    Quiz:
    {quiz_text}
    """
    importance_history_temp = [{"role": "user", "parts": [{"text": importance_prompt}]}]

    payload = {
        "contents": importance_history_temp,
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 300
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            important_questions = result["candidates"][0]["content"].get("parts")[0].get("text", "Could not identify important questions.")
            return important_questions
        else:
            return "Could not identify important questions. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        return f"Error identifying important questions: {e}"
    except Exception as e:
        return f"An unexpected error occurred while identifying important questions: {e}"

# NEW: Identify Important Points from Summary
async def identify_important_points_from_summary(summary_text: str):
    """Identifies and extracts important points from a given summary text using Gemini."""
    print("🧠 Identifying important points from the summary...")
    importance_prompt = f"""
    From the following summary, extract the 3 to 5 most crucial key points.
    Present them as a numbered list.

    Summary:
    {summary_text}
    """
    importance_history_temp = [{"role": "user", "parts": [{"text": importance_prompt}]}]

    payload = {
        "contents": importance_history_temp,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 200
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            important_points = result["candidates"][0]["content"].get("parts")[0].get("text", "Could not identify important points.")
            return important_points
        else:
            return "Could not identify important points. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        return f"Error identifying important points: {e}"
    except Exception as e:
        return f"An unexpected error occurred while identifying important points: {e}"

# NEW: Generate Important Questions from Notes
async def generate_important_questions_from_notes(notes_content: str):
    """Generates important questions directly from the full notes content using Gemini."""
    print("🧠 Generating important questions directly from the notes...")
    questions_prompt = f"""
    You are an educational assistant. Based on the following study notes, generate 3 to 5 comprehensive and important open-ended questions that a student should be able to answer after studying these notes. These questions should cover the most crucial concepts.
    Present them as a numbered list.

    Notes:
    {notes_content}
    """
    questions_history_temp = [{"role": "user", "parts": [{"text": questions_prompt}]}]

    payload = {
        "contents": questions_history_temp,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 400
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            important_questions = result["candidates"][0]["content"].get("parts")[0].get("text", "Could not generate important questions from notes.")
            return important_questions
        else:
            return "Could not generate important questions from notes. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        return f"Error generating important questions from notes: {e}"
    except Exception as e:
        return f"An unexpected error occurred while generating important questions from notes: {e}"

# NEW: Generate Practice Questions from Notes
async def generate_practice_questions(notes_content: str):
    """Generates general practice questions directly from the full notes content using Gemini."""
    print("🧠 Generating practice questions directly from the notes...")
    questions_prompt = f"""
    You are an educational assistant. Based on the following study notes, generate 3 to 5 comprehensive practice questions that cover various aspects of the material. These should be good questions for a student to use for self-study.
    Present them as a numbered list.

    Notes:
    {notes_content}
    """
    questions_history_temp = [{"role": "user", "parts": [{"text": questions_prompt}]}]

    payload = {
        "contents": questions_history_temp,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 400
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            practice_questions = result["candidates"][0]["content"].get("parts")[0].get("text", "Could not generate practice questions from notes.")
            return practice_questions
        else:
            return "Could not generate practice questions from notes. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        return f"Error generating practice questions from notes: {e}"
    except Exception as e:
        return f"An unexpected error occurred while generating practice questions from notes: {e}"


# --- New Feature: Content Summarization ---
async def summarize_content(text_to_summarize: str):
    """Summarizes provided text content using Gemini."""
    global last_generated_summary

    print("📚 Summarizing the notes...")
    summary_prompt = f"""
    Please provide a concise and informative summary of the following text using a numbered list (1., 2., 3., etc.).
    Each numbered point should highlight a main point or key takeaway.

    Text:
    {text_to_summarize}
    """
    summary_history_temp = [{"role": "user", "parts": [{"text": summary_prompt}]}]

    payload = {
        "contents": summary_history_temp,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 500
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            summary_text = result["candidates"][0]["content"].get("parts")[0].get("text", "No summary generated.")
            last_generated_summary = summary_text
            return summary_text
        else:
            last_generated_summary = None
            return "Could not generate summary. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        last_generated_summary = None
        return f"Error summarizing content: {e}"
    except Exception as e:
        last_generated_summary = None
        return f"An unexpected error occurred during summarization: {e}"

# --- Document Chunking and Retrieval (for Local Model) ---
def chunk_text(text, chunk_size=500, overlap=50):
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

    if not relevant_chunks:
        return chunks[0] if chunks else ""
    
    return "\n---\n".join(relevant_chunks)

# --- Main Processing Function for UI Interaction ---
async def process_user_request(user_input_text: str):
    """
    Processes a single user request from the UI, routes it to the appropriate
    AI model or command handler, and returns the assistant's response.
    """
    global science_notes_content, chat_history, loaded_file_name, science_notes_chunks, last_generated_quiz, last_generated_summary

    assistant_response = ""
    
    # --- NEW: Detect Emotion after transcription ---
    emotion_result = detect_emotion(user_input_text)
    # Print emotion to console for debugging, but UI can also display it
    print(f"Emotion detected in your input: {emotion_result['label']} (Score: {emotion_result['score']})")
    # You can add logic here to tailor the assistant's response based on emotion
    # e.g., if emotion_result['label'] == 'frustration': assistant_response = "I sense you might be frustrated. How can I help clarify things?"

    # --- Command Handling ---
    translator = str.maketrans('', '', string.punctuation)
    cleaned_input_text = user_input_text.translate(translator).strip().lower()

    command = cleaned_input_text

    if cleaned_input_text in ["size notes", "summarise notes", "summary notes", "sum rise notes", "samarize notes"]:
        command = "summarize notes"
    elif cleaned_input_text in ["important questions from quiz", "show important questions from quiz", "important quiz questions"]:
        command = "important questions from quiz"
    elif cleaned_input_text in ["important points from summary", "important summary points", "key points from summary"]:
        command = "important points from summary"
    elif cleaned_input_text in ["important questions from notes", "important questions from text", "key questions from notes"]:
        command = "important questions from notes"
    elif cleaned_input_text in ["practice questions", "generate practice questions", "give me practice questions", "study questions"]:
        command = "practice questions"
    elif cleaned_input_text in ["show history", "view history", "display history", "chat history"]:
        command = "show history"
    elif cleaned_input_text.startswith("load notes"):
        # Special handling for 'load notes <filepath>' command
        command_parts = user_input_text.split(' ', 1) # Split only once to get "load notes" and the rest as path
        if len(command_parts) > 1:
            command = "load notes"
            file_path_from_ui = command_parts[1].strip().strip('"') # Get path and remove potential quotes
        else:
            command = "invalid_load_notes_format" # Indicate malformed command
            file_path_from_ui = "" # Initialize to avoid error


    print(f"DEBUG: Processed command: '{command}' from input: '{user_input_text}'")

    if command == 'clear chat':
        chat_history = []
        assistant_response = "Conversational memory cleared."
    elif command == 'show history':
        if chat_history:
            assistant_response = "Here is our conversation history:\n"
            for entry in chat_history:
                role = entry['role'].capitalize()
                if 'text' in entry['parts'][0]:
                    content = entry['parts'][0]['text']
                elif 'functionCall' in entry['parts'][0]:
                    func_call = entry['parts'][0]['functionCall']
                    content = f"Tool Call: {func_call.get('name')}({func_call.get('args')})"
                elif 'functionResponse' in entry['parts'][0]:
                    func_resp = entry['parts'][0]['functionResponse']
                    content = f"Tool Response ({func_resp.get('name')}): {func_resp.get('response', {}).get('text', 'No response text.')[:100]}..."
                else:
                    content = "Unsupported history entry format."
                assistant_response += f"{role}: {content}\n"
        else:
            assistant_response = "The chat history is currently empty."
    elif command == 'quiz me':
        if science_notes_content:
            quiz_topic = "Your Notes"
            if loaded_file_name:
                base_name = os.path.splitext(loaded_file_name)[0]
                quiz_topic = base_name.replace('_', ' ').replace('-', ' ').title()
            quiz_output = await generate_quiz(quiz_topic, science_notes_content)
            assistant_response = "Here's your quiz:\n" + quiz_output
        else:
            assistant_response = "Please load notes first using 'load notes' command."
    elif command == 'important questions from quiz':
        if last_generated_quiz:
            important_q_output = await identify_important_questions_from_quiz(last_generated_quiz)
            assistant_response = "Here are some important questions from the last quiz:\n" + important_q_output
        else:
            assistant_response = "No quiz has been generated yet. Please use 'quiz me' first."
    elif command == 'important points from summary':
        if last_generated_summary:
            important_p_output = await identify_important_points_from_summary(last_generated_summary)
            assistant_response = "Here are some important points from the last summary:\n" + important_p_output
        else:
            assistant_response = "No summary has been generated yet. Please use 'summarize notes' first."
    elif command == 'important questions from notes':
        if science_notes_content:
            important_q_notes_output = await generate_important_questions_from_notes(science_notes_content)
            assistant_response = "Here are some important questions directly from your notes:\n" + important_q_notes_output
        else:
            assistant_response = "Please load notes first using 'load notes' command."
    elif command == 'practice questions':
        if science_notes_content:
            practice_q_output = await generate_practice_questions(science_notes_content)
            assistant_response = "Here are some practice questions from your notes:\n" + practice_q_output
        else:
            assistant_response = "Please load notes first using 'load notes' command."
    elif command == 'summarize notes':
        if science_notes_content:
            summary_output = await summarize_content(science_notes_content)
            assistant_response = "Here's a summary of the notes:\n" + summary_output
        else:
            assistant_response = "Please load notes first using 'load notes' command."
    elif command == 'load notes': # This branch is now specifically for the parsed 'load notes <filepath>'
        print(f"DEBUG: Attempting to load notes from path extracted: '{file_path_from_ui}'")
        new_content = read_document_content_from_file(file_path_from_ui)
        if new_content:
            science_notes_content = new_content
            loaded_file_name = os.path.basename(file_path_from_ui)
            science_notes_chunks = chunk_text(science_notes_content)
            assistant_response = f"Successfully loaded notes from '{file_path_from_ui}'. Ready to answer questions about this document!"
            chat_history = [] # Clear chat history when new notes are loaded
        else:
            assistant_response = f"Failed to load notes from '{file_path_from_ui}'. Please check the path and file permissions, or ensure it's not a scanned PDF."
            science_notes_content = None
            loaded_file_name = None
            science_notes_chunks = []
    elif command == 'invalid_load_notes_format':
        assistant_response = "Invalid 'load notes' command format. Please use 'load notes <filepath>' (e.g., 'load notes C:/path/to/file.txt')."
    elif command == 'exit':
        assistant_response = "Exiting assistant. Goodbye!"
    # --- Q&A Routing Logic (General questions not matching specific commands) ---
    else:
        if science_notes_content:
            print("🧠 Routing to local model for contextual Q&A with loaded notes (using chunks)...")
            relevant_context = find_relevant_chunks(user_input_text, science_notes_chunks)
            
            if relevant_context:
                assistant_response = get_local_answer(user_input_text, context_text=relevant_context)
                # If local model can't answer well from context, try Gemini
                if "state that" in assistant_response.lower() or "not in the response" in assistant_response.lower() or "sorry" in assistant_response.lower() or "i cannot answer" in assistant_response.lower() or "could not find the answer" in assistant_response.lower():
                    print("🧠 Local model couldn't answer from context. Routing to Gemini for general knowledge/search...")
                    assistant_response = await get_gemini_answer(user_input_text)
            else:
                print("🧠 No relevant chunks found in the document. Routing to Gemini for general knowledge/search...")
                assistant_response = await get_gemini_answer(user_input_text)
        else:
            print("🌐 Routing to Gemini for general Q&A / web search (no notes loaded)...")
            assistant_response = await get_gemini_answer(user_input_text)
    
    print("🤖 Assistant Answer:", assistant_response)
    return assistant_response # Return the response for the UI

# --- Canvas Environment Entry Point ---
# This block will be executed by the Canvas environment when the UI makes a request.
if __name__ == "__main__":
    try:
        # Read the entire incoming JSON payload from standard input
        input_data_str = sys.stdin.read()
        
        # Check if input_data_str is empty (can happen on initial load or if no body is sent)
        if not input_data_str:
            # This might be an initial load, or an empty request.
            # We can return an initial greeting or just an empty string.
            print("Hello! How can I help you today?")
            sys.exit(0) # Exit gracefully after initial greeting

        # Parse the JSON string into a Python dictionary
        input_payload = json.loads(input_data_str)

        user_input_text = input_payload.get('user_input', '')
        input_mode = input_payload.get('input_mode', 'text')

        # Run the asynchronous processing function
        response_from_assistant = asyncio.run(process_user_request(user_input_text))

        # Print the final response to standard output.
        # This is what the JavaScript's `response.text()` will receive.
        print(response_from_assistant)

    except json.JSONDecodeError:
        print("Error: Invalid JSON input received from UI. Make sure the UI sends valid JSON.")
        sys.exit(1) # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred in the Python backend: {e}")
        sys.exit(1) # Exit with error code