# Smart Classroom Assistant: Interactive Demo & Code Walkthrough

This document provides a guided tour through the **core functionalities** and **code implementation** of the Smart Classroom Assistant. It is structured like a simplified Jupyter Notebook to help you understand the concepts alongside code snippets.

---

## ✨ Table of Contents

1. [Project Setup and Core Utilities](#project-setup-and-core-utilities)
   - [Key Imports](#key-imports)
   - [API Key Configuration](#api-key-configuration)
2. [Document Processing: Loading and Chunking Notes](#document-processing-loading-and-chunking-notes)
   - [Reading Document Content](#reading-document-content)
   - [Text Chunking and Relevant Chunk Retrieval](#text-chunking-and-relevant-chunk-retrieval)
3. [Local AI Models: Optimized with OpenVINO](#local-ai-models-optimized-with-openvino)
   - [Loading Flan-T5 for QA](#loading-flan-t5-for-qa)
   - [Loading Emotion Classifier](#loading-emotion-classifier)
4. [Cloud AI Services: Gemini API with Tool Use](#cloud-ai-services-gemini-api-with-tool-use)
   - [Real-time Search Tool](#real-time-search-tool)
   - [Getting Answers from Gemini](#getting-answers-from-gemini)
5. [Educational Features: Quiz Generation & Summarization](#educational-features-quiz-generation--summarization)
   - [Generating Quizzes](#generating-quizzes)
   - [Summarizing Content](#summarizing-content)
6. [Voice Assistant Utility (Console-based)](#voice-assistant-utility-console-based)
   - [Recording Audio](#recording-audio)
   - [Transcribing Audio](#transcribing-audio)
   - [Speaking Answers with pyttsx3](#speaking-answers-with-pyttsx3)
7. [Putting It All Together: The Streamlit App Flow](#putting-it-all-together-the-streamlit-app-flow)
8. [Conclusion](#conclusion)

---

## 1. Project Setup and Core Utilities

### 1.1. Key Imports

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
````

---

### 1.2. API Key Configuration

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "YOUR_GOOGLE_CSE_API_KEY_HERE")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "YOUR_GOOGLE_CSE_ID_HERE")
```

---

## 2. Document Processing: Loading and Chunking Notes

### 2.1. Reading Document Content

Supports `.txt`, `.pdf`, and `.docx`:

```python
def read_document_content_from_file(file_path):
    ...
```

Example:

```python
notes_content = read_document_content_from_file("science_notes.txt")
```

---

### 2.2. Text Chunking and Relevant Chunk Retrieval

#### Chunking:

```python
def chunk_text(text, chunk_size=400, overlap=50):
    ...
```

#### Finding Relevant Chunks:

```python
def find_relevant_chunks(question, chunks, top_n=2):
    ...
```

---

## 3. Local AI Models: Optimized with OpenVINO

### 3.1. Loading Flan-T5 for QA

```python
def load_local_qa_generator_ov():
    ...
```

#### Generating Answers:

```python
def get_local_answer(question_text, context_text=None):
    ...
```

---

### 3.2. Loading Emotion Classifier

```python
def load_emotion_classifier_ov():
    ...
```

#### Detecting Emotion:

```python
def detect_emotion(text: str):
    ...
```

---

## 4. Cloud AI Services: Gemini API with Tool Use

### 4.1. Real-time Search Tool

```python
async def search_tool(query: str) -> str:
    ...
```

---

### 4.2. Getting Answers from Gemini

```python
async def get_gemini_answer(question_text: str):
    ...
```

---

## 5. Educational Features: Quiz Generation & Summarization

### 5.1. Generating Quizzes

```python
async def generate_quiz(topic: str, context: str):
    ...
```

---

### 5.2. Summarizing Content

```python
async def summarize_content(text_to_summarize: str):
    ...
```

---

## 6. Voice Assistant Utility (Console-based)

### 6.1. Recording Audio

```python
def record_audio(duration_seconds=5, samplerate=16000):
    ...
```

---

### 6.2. Transcribing Audio

```python
def transcribe_audio(audio_file_path):
    ...
```

---

### 6.3. Speaking Answers with pyttsx3

```python
def speak_answer(answer_text):
    ...
```

---

## 7. Putting It All Together: The Streamlit App Flow

* File uploader in sidebar for notes.
* Chat display for conversation history.
* User input routed to local model first for contextual Q\&A.
* Fallback to Gemini if local model fails or no relevant context is found.
* Educational commands like **"summarize notes"** or **"quiz me"** are handled dynamically.

---

## 8. Conclusion

This interactive walkthrough demonstrates the **modular design** and **key functionalities** of the Smart Classroom Assistant. By combining **local optimized AI models** with **powerful cloud services** and an intuitive **Streamlit interface**, the project delivers a versatile tool to enhance educational experiences.

---

### 🚀 **Next Steps**

* Integrate vector store-based semantic search for improved chunk retrieval.
* Enhance voice assistant with real-time Whisper and pyttsx3 pipelines.
* Extend quiz generator to support difficulty levels and Bloom’s taxonomy tagging.

---

> **Author:** Harshit
> **Project:** Smart Classroom Assistant
> **Date:** July 2025
