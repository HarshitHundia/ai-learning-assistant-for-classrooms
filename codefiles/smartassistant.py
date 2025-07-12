import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyttsx3
import requests
import json
import os # Import os for environment variables (if needed for API key, though we'll keep it empty for Canvas)

# -------------------------------
# 1️⃣  RECORD VOICE INPUT
# -------------------------------
def record_audio(duration_seconds=5, samplerate=16000):
    """Records audio input from the microphone."""
    print("🎤 Recording... Speak now")
    try:
        # Record audio
        recording = sd.rec(int(duration_seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait() # Wait until recording is finished

        # Normalize and save to WAV file
        recording = np.squeeze(recording)
        # Ensure the recording is not empty before normalizing
        if np.max(np.abs(recording)) > 0:
            recording = (recording / np.max(np.abs(recording)) * 32767).astype(np.int16)
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
        whisper_model = whisper.load_model("base")
        # Ensure FP16 warning is handled if it appears, though it's just a warning.
        question = whisper_model.transcribe(audio_file_path)["text"].strip()
        print("📝 You said:", question)
        return question
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

# -------------------------------
# 3️⃣  ANSWER USING GEMINI API
# -------------------------------
async def get_gemini_answer(question_text):
    """Fetches an answer from the Gemini 2.0 Flash API."""
    print("🤔 Getting answer from Gemini (please wait)...")
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": question_text}]})

    payload = {"contents": chat_history}
    # IMPORTANT: The API key is automatically provided by the Canvas environment when left empty.
    apiKey = "AIzaSyAgrpayn0eNJy-hqm8OrOOGf6mFrzyaGPE"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

    try:
        # Using requests for synchronous fetch in Python. In a browser context, `fetch` is used.
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        else:
            print("❗ Warning: Gemini API response structure unexpected or content missing.")
            print("Full response:", result) # Print full response for debugging
            return "Sorry, I couldn't get an answer from Gemini."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Gemini API: {e}")
        return "Sorry, I'm having trouble connecting to the answer service."
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return "An unexpected error occurred while getting the answer."

# -------------------------------
# 4️⃣  SPEAK USING LOCAL pyttsx3
# -------------------------------
def speak_answer(answer_text):
    """Speaks the given text using pyttsx3."""
    print("🤖 Assistant Answer:", answer_text)
    try:
        engine = pyttsx3.init()
        engine.say(answer_text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Error during speech synthesis: {e}")

# Main execution flow
if __name__ == "__main__":
    import asyncio # Import asyncio to run the async function

    # 1. Record voice input
    audio_file = record_audio()
    if audio_file is None:
        print("Exiting due to audio recording issue.")
        exit()

    # 2. Transcribe using local Whisper
    question_text = transcribe_audio(audio_file)
    if question_text is None:
        print("Exiting due to transcription issue.")
        exit()

    # 3. Answer using Gemini API (running the async function)
    # The asyncio.run() function runs the given coroutine until it finishes.
    answer_text = asyncio.run(get_gemini_answer(question_text))
    if answer_text is None:
        print("Exiting due to answer generation issue.")
        exit()

    # 4. Speak using local pyttsx3
    speak_answer(answer_text)
