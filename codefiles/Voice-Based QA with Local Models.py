import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyttsx3
import threading # For interrupting speech
import time      # For short sleeps
from transformers import pipeline # For the local LLM
import os # For file path operations

# Global variables for the pyttsx3 engine and a flag to control speaking
global_engine = None
speaking_thread = None
stop_flag = threading.Event()

# -------------------------------
# Initialize the local Flan-T5 model
# -------------------------------
print("⏳ Loading local language model (google/flan-t5-base)... This may take a moment the first time.")
try:
    local_qa_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    print("✅ Local language model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local model: {e}")
    print("Please ensure you have 'transformers' installed and an active internet connection for the initial download.")
    # Exit if the model can't be loaded, as the core functionality depends on it.
    exit()

# -------------------------------
# 1️⃣  RECORD VOICE INPUT
# -------------------------------
def record_audio(duration_seconds=5, samplerate=16000):
    """Records audio input from the microphone."""
    print("🎤 Recording... Speak now")
    try:
        # Record audio
        recording = sd.rec(int(duration_seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait() # Wait for the recording to finish

        # Normalize and save the recording
        recording = np.squeeze(recording) # Remove single-dimensional entries from the shape of an array
        if np.max(np.abs(recording)) > 0: # Check if there's actual audio data
            recording = (recording / np.max(np.abs(recording)) * 32767).astype(np.int16) # Normalize to 16-bit PCM
        else:
            print("❗ Warning: Recorded audio was silent or too quiet.")
            return None

        file_name = "question.wav"
        wav.write(file_name, samplerate, recording)
        print(f"✅ Recording saved to {file_name}.")
        return file_name
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return None

# -------------------------------
# 2️⃣  TRANSCRIBE USING LOCAL WHISPER
# -------------------------------
def transcribe_audio(audio_file_path):
    """Transcribes audio using the local Whisper model."""
    try:
        print("🎧 Transcribing audio with Whisper...")
        # Load the Whisper model. It will download the model the first time.
        # "base" model is a good balance for speed and accuracy locally.
        whisper_model = whisper.load_model("base")
        question = whisper_model.transcribe(audio_file_path)["text"].strip()
        print("📝 You said:", question)
        return question
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        print("Make sure you have `openai-whisper` installed (`pip install openai-whisper`) and ffmpeg set up.")
        return None

# -------------------------------
# 3️⃣  READ DOCUMENT CONTENT
# -------------------------------
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

# -------------------------------
# 4️⃣  ANSWER USING LOCAL FLAN-T5 (with contextual support)
# -------------------------------
def get_local_model_answer(question_text, context_text=None):
    """
    Fetches an answer from a local Flan-T5 model, optionally using provided context.
    """
    print("🤔 Generating answer with local model (please wait)...")
    try:
        # Construct the prompt based on whether context is provided
        if context_text:
            # Explicitly instruct the model to use the context for a comprehensive answer.
            # Adding "Extract the full relevant sentence:" to push for complete extraction.
            prompt = f"Extract the full relevant sentence to answer the following question based ONLY on the provided context. Context: {context_text}\n\nQuestion: {question_text}"
        else:
            # If no context, it acts as a general Q&A (accuracy may vary, as discussed).
            prompt = f"Answer this question clearly and accurately: {question_text}"

        # Generate the answer. max_length controls the length of the response.
        # Increased max_length to allow for more comprehensive answers.
        answer_output = local_qa_generator(prompt, max_length=256) # Increased max_length
        answer = answer_output[0]['generated_text'].strip()
        return answer
    except Exception as e:
        print(f"❌ Error during local model inference: {e}")
        return "Sorry, I encountered an error while trying to generate an answer locally."

# -------------------------------
# 5️⃣  SPEAK USING LOCAL pyttsx3 (with threading for interruption)
# -------------------------------
def _speak_in_thread(text_to_speak):
    """Helper function to run speech synthesis in a separate thread."""
    global global_engine
    if global_engine is None:
        global_engine = pyttsx3.init()
        # You can set properties like voice, rate, volume here if needed
        # voices = global_engine.getProperty('voices')
        # global_engine.setProperty('voice', voices[10].id) # Example: set to a female voice

    stop_flag.clear() # Reset the stop flag for this new speech

    try:
        global_engine.say(text_to_speak)
        while global_engine.isBusy():
            if stop_flag.is_set():
                global_engine.stop()
                print("🔊 Speech forcefully stopped.")
                break
            time.sleep(0.1) # Check every 100ms for stop signal
        global_engine.runAndWait() # Ensure any buffered speech is played or stopped
    except Exception as e:
        print(f"❌ Error during speech synthesis in thread: {e}")

def speak_answer(answer_text):
    """Starts speaking the given text in a new thread."""
    global speaking_thread
    # If a previous speech thread is still active, stop it first.
    if speaking_thread and speaking_thread.is_alive():
        print("🔊 A speech is already in progress. Stopping it...")
        stop_speaking()
        speaking_thread.join() # Wait for the old thread to finish stopping

    print("🤖 Assistant Answer:", answer_text)
    # Create and start a new thread for speaking.
    speaking_thread = threading.Thread(target=_speak_in_thread, args=(answer_text,))
    speaking_thread.start()

def stop_speaking():
    """Signals the speaking thread to stop."""
    global global_engine, speaking_thread
    if speaking_thread and speaking_thread.is_alive():
        stop_flag.set() # Set the event to signal stopping to the thread
        if global_engine and global_engine.isBusy():
            global_engine.stop() # Force the pyttsx3 engine to stop immediately
        print("🔊 Attempted to stop speech.")
    else:
        print("🔊 No speech currently active to stop.")

# Main execution flow
if __name__ == "__main__":
    import sys
    # For Windows-specific non-blocking input for 's' to stop speech.
    # For Linux/macOS, a more complex solution (e.g., curses or threading with input())
    # would be needed for truly non-blocking key presses.
    if sys.platform == "win32":
        import msvcrt
    else:
        print("Note: 's' to stop speech only works reliably on Windows in this script due to msvcrt dependency.")

    print("\n--- Voice Assistant (Local Model - Contextual) Ready ---")
    print("Using Whisper for STT, google/flan-t5-base for QA, pyttsx3 for TTS.")
    print("This assistant can answer questions based on a text file you provide.")
    print("Disclaimer: For general factual questions *without* context, local models like Flan-T5-base")
    print("may provide less accurate answers compared to cloud APIs like Gemini.")
    print("-----------------------------------------------------------\n")

    document_content = None # Initialize document content to None

    while True:
        # Prompt for document path if not already loaded
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

        print("\n--- Listening for your question ---")
        audio_file = record_audio()
        if audio_file is None:
            print("Audio recording failed, please try again.")
            continue

        question_text = transcribe_audio(audio_file)
        if question_text is None or not question_text.strip():
            print("Transcription failed or no speech detected, please try again.")
            continue

        # Check for control commands spoken by the user
        if question_text.lower().strip() == 'exit':
            print("Exiting module. Goodbye!")
            break
        elif question_text.lower().strip() == 'new document' or question_text.lower().strip() == 'new doc':
            document_content = None # Reset document content to prompt for new file
            print("\n--- Loading a new document ---")
            continue

        # Get answer using the local model, passing the document content as context
        answer_text = get_local_model_answer(question_text, context_text=document_content)
        if answer_text is None:
            print("Answer generation failed, please try again.")
            continue

        speak_answer(answer_text)

        print("\n(Speaking... For Windows, press 's' and Enter to stop, then Enter again for next question)")
        # Simplified "stop" mechanism for demonstration.
        # This part still largely relies on msvcrt for true non-blocking input on Windows.
        # For cross-platform, a separate input thread or more advanced GUI is needed.
        while speaking_thread and speaking_thread.is_alive():
            if sys.platform == "win32":
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 's':
                        stop_speaking()
                        break # Exit this inner loop, let the main loop continue
            else:
                # Basic placeholder for non-Windows, direct input() would block
                pass
            time.sleep(0.1) # Small delay to not consume 100% CPU

        print("\n--- Finished ---")
        if speaking_thread and speaking_thread.is_alive():
            speaking_thread.join() # Ensure the speaking thread fully completes (or stops)

        print("\nReady for next question (or 'new document' / 'exit').\n")
