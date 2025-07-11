import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import json
import asyncio
import os
from transformers import pipeline # For the local Flan-T5 model
import string # Import string module for punctuation
import pypdf # For reading PDF files
from docx import Document # For reading .docx files

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

# --- Real-time Search API Configuration (NEW) ---
# You MUST replace these with your actual Google Custom Search API Key and Search Engine ID (CX)
# Get them from:
# 1. Google Cloud Console for API Key: https://console.cloud.google.com/apis/credentials
# 2. Programmable Search Engine for CX: https://programmablesearchengine.google.com/controlpanel/all
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
    exit()

# Global variable for conversational memory
chat_history = []
# Global variable for custom document content
science_notes_content = None # Will be loaded from a file
loaded_file_name = None # To store the name of the currently loaded file
science_notes_chunks = [] # Stores chunks of the loaded document for local model
last_generated_quiz = None # Stores the text of the last generated quiz
last_generated_summary = None # Stores the text of the last generated summary

# --- External Tool: Google Search (Now uses real API) ---
async def search_tool(query: str) -> str:
    """
    Performs a real-time web search for the given query using Google Custom Search JSON API.
    Requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to be configured.
    """
    # This check is now primarily for debugging/warning if keys are still placeholders
    if GOOGLE_CSE_API_KEY == "YOUR_GOOGLE_CSE_API_KEY" or GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID":
        print("❗ Warning: Google Custom Search API keys are not configured. Using simulated fallback.")
        # Fallback to simulated responses if API keys are not set
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

    # --- Real API Call ---
    search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"

    print(f"🔍 Performing real web search for: '{query}' using Google Custom Search API...")
    try:
        response = requests.get(search_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        search_results = response.json()

        if search_results and 'items' in search_results:
            snippets = []
            for item in search_results['items'][:3]: # Get top 3 snippets for brevity
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


# --- 1️⃣ RECORD VOICE INPUT ---
def record_audio(duration_seconds=5, samplerate=16000):
    """Records audio input from the microphone."""
    print("🎤 Recording... Speak now")
    try:
        # Provide visual feedback/countdown
        for i in range(duration_seconds, 0, -1):
            print(f"Recording in {i}...", end='\r')
            sd.sleep(1) # Sleep for 1 second (1000 milliseconds)
        print("Recording complete!   ") # Clear countdown message

        recording = sd.rec(int(duration_seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()

        recording = np.squeeze(recording)
        if np.max(np.abs(recording)) > 0:
            recording = (recording / np.max(np.abs(recording)) * 32767).astype(np.int16)
        else:
            print("❗ Warning: Recorded audio was silent or too quiet. Please try speaking louder or checking your microphone.")
            return None

        file_name = "question.wav"
        wav.write(file_name, samplerate, recording)
        print(f"✅ Recording saved to {file_name}.")
        return file_name
    except Exception as e:
        print(f"❌ Error during recording: {e}. Make sure your microphone is connected and drivers are installed.")
        return None

# --- 2️⃣ TRANSCRIBE USING LOCAL WHISPER ---
def transcribe_audio(audio_file_path):
    """
    Transcribes audio using the local Whisper model.
    Using a larger model and providing an initial prompt for better accuracy.
    Returns the raw transcription for external cleaning.
    """
    try:
        print("🎧 Transcribing audio with Whisper (using 'medium' model for improved accuracy)...")
        whisper_model = whisper.load_model("medium")
        
        initial_prompt = "Summarize notes, quiz me, load notes, clear chat, exit, cell structure, biology, prokaryotic, eukaryotic, nucleus, mitochondria, cricket, teams, rules, formats."
        
        # Get raw transcription from Whisper. No cleaning here.
        raw_transcription = whisper_model.transcribe(audio_file_path, language="en", initial_prompt=initial_prompt)["text"]
        print("📝 You said:", raw_transcription)
        
        return raw_transcription
    except Exception as e:
        print(f"❌ Error during transcription: {e}. Is the Whisper model downloaded correctly? Do you have enough RAM?")
        return None

# --- Custom Document Content Loading (Now supports .txt, .pdf, .docx) ---
def read_document_content_from_file(file_path):
    """
    Reads the content of a text file, PDF, or DOCX and returns it as a string.
    Handles basic file reading errors and informs about OCR limitation for PDFs.
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
                    return None # Return None if no text could be extracted
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
        if context_text:
            # Emphasize using ONLY the context for factual questions within the document
            prompt = f"Provide a comprehensive answer to the question based ONLY on the provided context. Context: {context_text}\n\nQuestion: {question_text}"
        else:
            # If no context, it acts as a general Q&A (accuracy may vary).
            prompt = f"Answer this question clearly and accurately: {question_text}"

        # Increased max_length to give more room for comprehensive answers
        # Removed temperature here as it's better controlled in Gemini, and local models can be sensitive.
        answer_output = local_qa_generator(prompt, max_length=256) 
        answer = answer_output[0]['generated_text'].strip()
        return answer
    except Exception as e:
        print(f"❌ Error during local model inference: {e}")
        return "Sorry, I encountered an error while trying to generate an answer locally."

# --- 3️⃣ GEMINI API Answer Generation (General QA, Tool Use, Conversational Memory) ---
async def get_gemini_answer(question_text: str):
    """
    Fetches an answer from the Gemini 2.0 Flash API, incorporating tool use
    for real-time information and conversational memory.
    """
    global chat_history # Use the global chat history

    # Append the current user question to chat history
    chat_history.append({"role": "user", "parts": [{"text": question_text}]})

    # Define the tool(s) to be used by Gemini
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
        "contents": chat_history, # Send the entire chat history for conversational memory
        "tools": tools,          # Include tool definitions
        "generationConfig": {
            "temperature": 0.9,  # Adjust temperature for creativity/randomness
            "maxOutputTokens": 800 # Allow for longer responses
        }
    }

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # Check for function call by the model
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("functionCall"):
            
            function_call = result["candidates"][0]["content"]["parts"][0]["functionCall"]
            function_name = function_call["name"]
            function_args = function_call["args"]

            print(f"🤖 Gemini decided to call a tool: {function_name} with args {function_args}")

            # Append model's tool call to history (this entry does not have 'text')
            chat_history.append({"role": "model", "parts": [{"functionCall": function_call}]})
            
            if function_name == "search_tool":
                tool_response_content = await search_tool(function_args.get("query"))
                
                # Corrected structure for functionResponse within parts
                chat_history.append({
                    "role": "tool",
                    "parts": [{"functionResponse": {"name": function_name, "response": {"text": tool_response_content}}}]
                })

                # Make a follow-up API call to Gemini with the tool's response
                payload_with_tool_response = {
                    "contents": chat_history,
                    "tools": tools, # Re-include tools for the follow-up turn
                    "generationConfig": payload["generationConfig"] # Keep same generation config
                }
                response_after_tool = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload_with_tool_response))
                response_after_tool.raise_for_status()
                result_after_tool = response_after_tool.json()

                if result_after_tool.get("candidates") and result_after_tool["candidates"][0].get("content") and \
                   result_after_tool["candidates"][0]["content"].get("parts"):
                    final_answer = result_after_tool["candidates"][0]["content"]["parts"][0]["text"]
                    chat_history.append({"role": "model", "parts": [{"text": final_answer}]}) # Add final answer to history
                    return final_answer
                else:
                    print("❗ Warning: Gemini API (after tool) response structure unexpected or content missing.")
                    print("Full response (after tool):", result_after_tool)
                    return "Sorry, I processed the search but couldn't get a clear answer from Gemini."
            else:
                return "Sorry, Gemini tried to call an unknown tool."

        # If no tool call, or if there's a direct answer from the first turn
        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            direct_answer = result["candidates"][0]["content"].get("parts")[0].get("text", "No text found in response.")
            chat_history.append({"role": "model", "parts": [{"text": direct_answer}]}) # Add direct answer to history
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
    global last_generated_quiz # Declare global to store the quiz

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
    # Use a fresh chat history for quiz generation to avoid conversational context interfering
    quiz_history = [{"role": "user", "parts": [{"text": quiz_prompt}]}]
    
    payload = {
        "contents": quiz_history,
        "generationConfig": {
            "temperature": 0.7, # Lower temperature for factual quiz generation
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
            last_generated_quiz = quiz_text # Store the generated quiz
            return quiz_text
        else:
            last_generated_quiz = None # Clear if generation failed
            return "Could not generate quiz. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        last_generated_quiz = None # Clear if generation failed
        return f"Error generating quiz: {e}"
    except Exception as e:
        last_generated_quiz = None # Clear if generation failed
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
    # Use a fresh chat history for this specific request
    importance_history = [{"role": "user", "parts": [{"text": importance_prompt}]}]

    payload = {
        "contents": importance_history,
        "generationConfig": {
            "temperature": 0.5, # Balance creativity and focus
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
    importance_history = [{"role": "user", "parts": [{"text": importance_prompt}]}]

    payload = {
        "contents": importance_history,
        "generationConfig": {
            "temperature": 0.3, # Focus on accuracy
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
    questions_history = [{"role": "user", "parts": [{"text": questions_prompt}]}]

    payload = {
        "contents": questions_history,
        "generationConfig": {
            "temperature": 0.6, # Balance coverage and relevance
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
    questions_history = [{"role": "user", "parts": [{"text": questions_prompt}]}]

    payload = {
        "contents": questions_history,
        "generationConfig": {
            "temperature": 0.7, # Slightly higher temperature for variety in practice questions
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
    global last_generated_summary # Declare global to store the summary

    print("📚 Summarizing the notes...")
    # Modified prompt to ask for numbered points
    summary_prompt = f"""
    Please provide a concise and informative summary of the following text using a numbered list (1., 2., 3., etc.).
    Each numbered point should highlight a main point or key takeaway.

    Text:
    {text_to_summarize}
    """
    # Use a fresh chat history for summarization
    summary_history = [{"role": "user", "parts": [{"text": summary_prompt}]}]

    payload = {
        "contents": summary_history,
        "generationConfig": {
            "temperature": 0.3, # Lower temperature for concise summarization
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
            last_generated_summary = summary_text # Store the generated summary
            return summary_text
        else:
            last_generated_summary = None # Clear if generation failed
            return "Could not generate summary. Gemini API response unexpected."
    except requests.exceptions.RequestException as e:
        last_generated_summary = None # Clear if generation failed
        return f"Error summarizing content: {e}"
    except Exception as e:
        last_generated_summary = None # Clear if generation failed
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
    question_keywords = set(word.lower() for word in question.split() if len(word) > 2) # Exclude short words
    
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_keywords = set(word.lower() for word in chunk.split() if len(word) > 2)
        # Calculate overlap of keywords
        score = len(question_keywords.intersection(chunk_keywords))
        if score > 0: # Only consider chunks that have at least one keyword match
            scored_chunks.append((score, i, chunk))
    
    # Sort by score in descending order
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Select top_n chunks
    relevant_chunks = [chunk for score, i, chunk in scored_chunks[:top_n]]

    if not relevant_chunks:
        # If no chunks match keywords, return the first chunk as a fallback
        # or indicate no relevant content found
        return chunks[0] if chunks else "" # Fallback to first chunk if available
    
    return "\n---\n".join(relevant_chunks) # Join with a separator


# --- Main Execution Flow ---
async def main_assistant_loop():
    global science_notes_content
    global chat_history
    global loaded_file_name
    global science_notes_chunks # Declare global for chunks
    global last_generated_quiz # Declare global for last quiz
    global last_generated_summary # NEW: Declare global for last summary

    print("\n--- Smart Classroom Assistant (Voice & Text-Only Output) ---")
    print("Features: Contextual Q&A (local Flan-T5), General Q&A (Gemini with Search Tool), Quiz Generation, Summarization, Conversational Memory.")
    print("Commands:")
    print("  'load notes': Load your notes file (supports .txt, .pdf, .docx).") # Updated description
    print("  'quiz me': Generate a quiz from the loaded notes.")
    print("  'important questions from quiz': Get important questions from the last generated quiz.")
    print("  'important points from summary': Get important points from the last generated summary.")
    print("  'important questions from notes': Get important questions directly from the notes file.")
    print("  'practice questions': Generate general practice questions from the notes.")
    print("  'summarize notes': Get a summary of the loaded notes.")
    print("  'clear chat': Reset conversational memory.")
    print("  'show history': Display recent chat history.")
    print("  'exit': Quit the assistant.")
    print("--------------------------------------------------\n")

    print("Welcome! Please load a notes file to begin, or proceed without notes for general Q&A.")
    initial_file_path = input("📂 Enter the path to your text/PDF/DOCX file (e.g., my_notes.txt), or type 'skip' to start without notes: ").strip() # Updated prompt
    if initial_file_path.lower() != 'skip':
        new_content = read_document_content_from_file(initial_file_path)
        if new_content:
            science_notes_content = new_content
            loaded_file_name = os.path.basename(initial_file_path)
            science_notes_chunks = chunk_text(science_notes_content) # Chunk the content
            print(f"Successfully loaded notes from '{initial_file_path}'.")
        else:
            print(f"Failed to load notes from '{initial_file_path}'. Starting without notes.")
            loaded_file_name = None
            science_notes_content = None # Ensure content is None if loading fails
            science_notes_chunks = []
    else:
        print("Starting without notes. You can load notes later using the 'load notes' command.")
        loaded_file_name = None
        science_notes_content = None
        science_notes_chunks = []

    while True:
        mode_choice = input("\nChoose input mode ('voice' or 'text', or 'exit'): ").strip().lower()
        if mode_choice == 'exit':
            print("Exiting assistant. Goodbye!")
            break

        question_text = ""
        if mode_choice == 'voice':
            audio_file = record_audio()
            if audio_file:
                question_text = transcribe_audio(audio_file)
            else:
                print("I didn't catch that. Please try again.")
                continue
        elif mode_choice == 'text':
            question_text = input("📝 Enter your question or command: ").strip()
        else:
            print("❗ Invalid mode. Please choose 'voice' or 'text'.")
            continue
        
        if not question_text:
            print("No input received. Please try again.")
            continue

        # --- Command Handling ---
        # Ensure question_text is cleaned for robust command matching
        translator = str.maketrans('', '', string.punctuation)
        cleaned_question_text = question_text.translate(translator).strip().lower()

        # Apply specific remapping for common misinterpretations of commands
        command = cleaned_question_text # Default to cleaned text

        if cleaned_question_text in ["size notes", "summarise notes", "summary notes", "sum rise notes", "samarize notes"]:
            command = "summarize notes"
        elif cleaned_question_text in ["important questions from quiz", "show important questions from quiz", "important quiz questions"]:
            command = "important questions from quiz"
        elif cleaned_question_text in ["important points from summary", "important summary points", "key points from summary"]:
            command = "important points from summary"
        elif cleaned_question_text in ["important questions from notes", "important questions from text", "key questions from notes"]:
            command = "important questions from notes"
        elif cleaned_question_text in ["practice questions", "generate practice questions", "give me practice questions", "study questions"]:
            command = "practice questions"
        elif cleaned_question_text in ["show history", "view history", "display history", "chat history"]:
            command = "show history"
        elif cleaned_question_text in ["load new notes", "change notes", "change file", "load document", "science notes", "load my notes"]:
            command = "load notes"


        print(f"DEBUG: Final command for matching (cleaned and potentially remapped): '{command}')")
        print(f"DEBUG: Repr of final command: '{repr(command)}'")


        assistant_response = ""
        # Store user input in chat history before processing
        # Note: get_gemini_answer also appends user/model roles, so this append
        # is primarily for direct commands not handled by get_gemini_answer's internal flow.
        # We ensure it's not double-added later.

        if command == 'clear chat':
            chat_history = []
            assistant_response = "Conversational memory cleared."
        elif command == 'show history':
            if chat_history:
                assistant_response = "Here is our conversation history:\n"
                for entry in chat_history:
                    role = entry['role'].capitalize()
                    
                    # Safely extract content based on its type
                    if 'text' in entry['parts'][0]:
                        content = entry['parts'][0]['text']
                    elif 'functionCall' in entry['parts'][0]:
                        func_call = entry['parts'][0]['functionCall']
                        content = f"Tool Call: {func_call.get('name')}({func_call.get('args')})"
                    elif 'functionResponse' in entry['parts'][0]:
                        func_resp = entry['parts'][0]['functionResponse']
                        # Assuming 'response' in functionResponse will always have 'text' for our simulated tool
                        content = f"Tool Response ({func_resp.get('name')}): {func_resp.get('response', {}).get('text', 'No response text.')[:100]}..." # Truncate long responses
                    else:
                        content = "Unsupported history entry format."
                        
                    assistant_response += f"{role}: {content}\n"
            else:
                assistant_response = "The chat history is currently empty."
            # No need to append assistant_response to chat_history here, as show history is just a display function
        elif command == 'quiz me':
            if science_notes_content:
                quiz_topic = "Your Notes"
                if loaded_file_name:
                    base_name = os.path.splitext(loaded_file_name)[0]
                    quiz_topic = base_name.replace('_', ' ').replace('-', ' ').title()

                print(f"DEBUG: Generating quiz on topic: '{quiz_topic}'")
                quiz_output = await generate_quiz(quiz_topic, science_notes_content)
                assistant_response = "Here's your quiz:\n" + quiz_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "Please load notes first using 'load notes' command."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'important questions from quiz': # Handle important questions from quiz
            if last_generated_quiz:
                important_q_output = await identify_important_questions_from_quiz(last_generated_quiz)
                assistant_response = "Here are some important questions from the last quiz:\n" + important_q_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "No quiz has been generated yet. Please use 'quiz me' first."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'important points from summary': # Handle important points from summary
            if last_generated_summary:
                important_p_output = await identify_important_points_from_summary(last_generated_summary)
                assistant_response = "Here are some important points from the last summary:\n" + important_p_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "No summary has been generated yet. Please use 'summarize notes' first."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'important questions from notes': # Handle important questions from notes
            if science_notes_content:
                important_q_notes_output = await generate_important_questions_from_notes(science_notes_content)
                assistant_response = "Here are some important questions directly from your notes:\n" + important_q_notes_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "Please load notes first using 'load notes' command."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'practice questions': # NEW: Handle practice questions
            if science_notes_content:
                practice_q_output = await generate_practice_questions(science_notes_content)
                assistant_response = "Here are some practice questions from your notes:\n" + practice_q_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "Please load notes first using 'load notes' command."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'summarize notes':
            if science_notes_content:
                summary_output = await summarize_content(science_notes_content)
                assistant_response = "Here's a summary of the notes:\n" + summary_output
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = "Please load notes first using 'load notes' command."
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
        elif command == 'load notes':
            print("\n--- Loading a new document ---")
            file_path = input("📂 Enter the path to your text/PDF/DOCX file (e.g., my_new_notes.txt): ").strip() # Updated prompt
            new_content = read_document_content_from_file(file_path)
            if new_content:
                science_notes_content = new_content
                loaded_file_name = os.path.basename(file_path)
                science_notes_chunks = chunk_text(science_notes_content) # Re-chunk the content
                assistant_response = f"Successfully loaded notes from '{file_path}'. Ready to answer questions about this document!"
                chat_history = [] # Clear chat history when new notes are loaded - already appended before this block
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            else:
                assistant_response = f"Failed to load notes from '{file_path}'. Please check the path and file permissions, or ensure it's not a scanned PDF." # Updated error message
                science_notes_content = None # Ensure content is None if loading fails
                loaded_file_name = None
                science_notes_chunks = []
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]}) # Manually append
            
        # --- Q&A Routing Logic (General questions not matching specific commands) ---
        else:
            # We already appended the user's question to chat_history in get_gemini_answer, so no need here.
            # However, if we're routing to local model first, we need to add the user input here.
            # The current structure has `chat_history.append({"role": "user", "parts": [{"text": question_text}]})`
            # at the beginning of `get_gemini_answer`, which is called *after* this routing logic.
            # This means for local model answers, the user input is not yet in chat_history.

            # Re-evaluating chat_history append for clarity and correctness:
            # The `chat_history.append({"role": "user", "parts": [{"text": question_text}]})` at the very beginning of main_assistant_loop,
            # before any command processing, is the most robust place for user input.
            # Then, each branch (command or Q&A) needs to ensure the model's response is appended.
            
            # Since `get_gemini_answer` handles its own user and model appends,
            # we need to ensure local_answer path *also* appends its response.

            if science_notes_content:
                # When a document is loaded, prioritize local model for contextual Q&A using chunks
                print("🧠 Routing to local model for contextual Q&A with loaded notes (using chunks)...")
                # Pass the original question_text to find relevant chunks
                relevant_context = find_relevant_chunks(question_text, science_notes_chunks)
                
                if relevant_context:
                    # Pass the original question and the relevant context to the local model
                    assistant_response = get_local_answer(question_text, context_text=relevant_context)
                    # For local answers, explicitly append the model response
                    chat_history.append({"role": "model", "parts": [{"text": assistant_response}]})
                    
                    # If local model says it cannot answer from text, then try Gemini
                    if "state that" in assistant_response.lower() or "not in the response" in assistant_response.lower() or "sorry" in assistant_response.lower() or "i cannot answer" in assistant_response.lower() or "could not find the answer" in assistant_response.lower():
                        print("🧠 Local model couldn't answer from context. Routing to Gemini for general knowledge/search...")
                        # get_gemini_answer will append its own user/model interactions
                        answer_text_from_gemini = await get_gemini_answer(question_text)
                        assistant_response = "The local model couldn't find the answer in the document, so here's a general answer:\n" + answer_text_from_gemini
                        # This response from get_gemini_answer is already appended by get_gemini_answer itself.
                        # No extra append here.
                else:
                    print("🧠 No relevant chunks found in the document. Routing to Gemini for general knowledge/search...")
                    # get_gemini_answer will append its own user/model interactions
                    assistant_response = await get_gemini_answer(question_text)
            else:
                # If no document is loaded, always use Gemini for general knowledge
                print("🌐 Routing to Gemini for general Q&A / web search (no notes loaded)...")
                # get_gemini_answer will append its own user/model interactions
                assistant_response = await get_gemini_answer(question_text)
        
        # This final append logic needs to be careful not to double-append.
        # Given the changes, each branch (command or Q&A) now explicitly appends.
        # So, no general append here.

        # --- Display Text Answer ---
        print("🤖 Assistant Answer:", assistant_response) # Directly print the response
        print("\n--------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main_assistant_loop())