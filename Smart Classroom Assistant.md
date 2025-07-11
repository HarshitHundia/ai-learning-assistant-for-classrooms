# 📚 Smart Classroom Assistant: Interactive Demo & Code Walkthrough

This document provides a guided tour through the core functionalities and code implementation of the **Smart Classroom Assistant**, structured like a simplified Jupyter Notebook to understand concepts alongside code snippets.

---

## 🛠️ 1. Project Setup and Core Utilities

### 1.1. Key Imports

We leverage several powerful libraries for AI, web interaction, and file processing.

```python
import streamlit as st
import requests
import json
import os

from optimum.intel.openvino import OVModelForSeq2SeqLM, OVModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoConfig

import pypdf
from docx import Document

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyttsx3
import threading
import time
import asyncio
```

### 1.2. API Key Configuration

Securely managing API keys is crucial.

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "YOUR_GOOGLE_CSE_API_KEY_HERE")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "YOUR_GOOGLE_CSE_ID_HERE")
```

---

## 📄 2. Document Processing: Loading and Chunking Notes

### 2.1. Reading Document Content

Supports `.txt`, `.pdf`, `.docx` files.

➡️ **Key Function:** `read_document_content_from_file(file_path)`

### 2.2. Text Chunking and Relevant Chunk Retrieval

➡️ **Functions:**

* `chunk_text(text, chunk_size=400, overlap=50)`
* `find_relevant_chunks(question, chunks, top_n=2)`

---

## 🤖 3. Local AI Models: Optimized with OpenVINO

### 3.1. Loading and Optimizing Flan-T5 for QA

➡️ **Functions:**

* `load_local_qa_generator_ov()`
* `get_local_answer(question_text, context_text)`

### 3.2. Loading and Optimizing Emotion Classifier

➡️ **Functions:**

* `load_emotion_classifier_ov()`
* `detect_emotion(text)`

---

## ☁️ 4. Cloud AI Services: Gemini API with Tool Use

### 4.1. Real-time Search Tool

➡️ **Function:** `search_tool(query)`

### 4.2. Getting Answers from Gemini (with Tool Integration)

➡️ **Function:** `get_gemini_answer(question_text)`

---

## 🎓 5. Educational Features: Quiz Generation & Summarization

### 5.1. Generating Quizzes

➡️ **Function:** `generate_quiz(topic, context)`

### 5.2. Summarizing Content

➡️ **Function:** `summarize_content(text_to_summarize)`

---

## 🎤 6. Voice Assistant Utility (Console-based)

### 6.1. Recording Audio

➡️ **Function:** `record_audio(duration_seconds=5, samplerate=16000)`

### 6.2. Transcribing Audio with Whisper

➡️ **Function:** `transcribe_audio(audio_file_path)`

### 6.3. Speaking Answers with pyttsx3

➡️ **Functions:** `speak_answer(answer_text)`, `stop_speaking()`

---

## 💻 7. Putting It All Together: The Streamlit App Flow

➡️ **Main script:** `smart_assistant_app_ui.py` orchestrates all components with:

* **Session state management**
* **File upload sidebar**
* **Chat display area**
* **User input handling**
* **Q\&A routing logic (local model + Gemini fallback)**

---

## 🚀 8. Conclusion

This interactive walkthrough demonstrates the **modular design and key functionalities** of the Smart Classroom Assistant. By combining **local optimized AI models, powerful cloud services, and an intuitive Streamlit interface**, the project delivers a versatile tool to enhance classroom learning.

---

✅ **Next Steps (Optional Future Work Section)**

* Integrate **student database for personalized learning**
* Expand to **multimodal (image-based) Q\&A**
* Deploy on **Intel hardware with OpenVINO for faster inference**
* Build an Android app interface for mobile classrooms

---
