import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

# Step 1: Record audio for 5 seconds
fs = 16000  # 16kHz sample rate
duration = 5  # seconds
print("🎤 Recording... Speak now")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
wav.write("student_question.wav", fs, recording)
print("✅ Recording saved.")
# Step 2: Load Whisper model and transcribe
model = whisper.load_model("base")  # you can also use "tiny" for faster performance
result = model.transcribe("student_question.wav")

print("\n📝 Transcribed Text:", result["text"])
