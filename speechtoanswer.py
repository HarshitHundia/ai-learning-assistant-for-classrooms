import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import wikipedia
from transformers import pipeline

# Step 1: Record audio
fs = 16000
duration = 5
print("🎤 Recording... Speak now")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
wav.write("question.wav", fs, audio)
print("✅ Recording saved.")

# Step 2: Transcribe
model = whisper.load_model("base")
question = model.transcribe("question.wav")["text"].strip()
print("📝 You said:", question)

# Step 3: Get context from Wikipedia
try:
    # Get last 1-3 important words from question
    topic_guess = question.split()[-1]
    context = wikipedia.summary(topic_guess, sentences=2)
except Exception as e:
    print("❌ Could not find topic on Wikipedia:", e)
    context = "France is a country in Europe. Its capital is Paris."

# Step 4: Use QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
answer = qa_model({
    "question": question,
    "context": context
})["answer"]

print("🤖 Assistant Answer:", answer)