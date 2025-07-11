```markdown
# Smart Classroom Assistant – AI-Powered Interactive Learning Platform

An interactive AI assistant that enhances classroom learning by **answering contextual questions from notes, generating quizzes, summarizing content, and detecting emotions**, using optimized local models with OpenVINO and Gemini cloud APIs.

---

## 🚀 **Project Overview**

This project implements a **Smart Classroom Assistant** capable of:

- 📄 Reading and processing documents (`.txt`, `.pdf`, `.docx`)
- 🤖 Answering questions contextually using local Flan-T5 (OpenVINO optimized)
- 🤖 Answering questions rather then the document (general questions using api)
- 📝 Generating multiple-choice quizzes for revision  
- ✨ Summarizing notes for quick study  
- 😊 Detecting emotions from user input  
- 🎤 Transcribing speech to text using Whisper  
- 💬 Speaking out answers using pyttsx3

All features are accessible through a **Streamlit web interface** for interactive and intuitive use.

---

## 🧠 **Key AI Models & APIs Used**

| Model/API | Purpose | Optimization |
| --- | --- | --- |
| **Flan-T5 Base** | Contextual Question Answering | OpenVINO optimized for CPU inference |
| **Gemini Flash API** | Cloud-based general Q&A, summarization, quiz generation | Google Generative AI |
| **DistilRoberta Emotion Classifier** | Emotion detection from text | OpenVINO optimized |
| **Whisper** | Speech-to-text transcription | Local execution |
| **pyttsx3** | Text-to-speech | Local execution |

---

## ⚙️ **Features**

✅ Read and process `.txt`, `.pdf`, `.docx` notes  
✅ Chunk text for efficient QA context  
✅ Find most relevant chunks for user queries  
✅ Answer questions using local models, with Gemini fallback  
✅ Generate quizzes for any topic  
✅ Summarize long notes to key points  
✅ Detect user's emotional sentiment  
✅ Voice interaction via Whisper + pyttsx3

---

## 📂 **Project Structure**

```

ai-learning-assistant-for-classrooms/
├── main\_smart\_assistant\_app\_ui.py
├── ov\_flan\_t5\_base/                # OpenVINO optimized Flan-T5 model files (if saved locally)
├── ov\_emotion\_classifier/          # OpenVINO optimized emotion model files (if saved locally)
├── requirements.txt
└── README.md

````

*(Adjust if your actual repo folders differ)*

---

## 💻 **Installation & Setup**

1. **Clone the repository**

```bash
git clone https://github.com/HarshitHundia/ai-learning-assistant-for-classrooms.git
cd ai-learning-assistant-for-classrooms
````

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup Environment Variables**

Create a `.env` file or export directly:

```
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CSE_API_KEY=your_google_cse_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
```

---

## ▶️ **Running the Application**

Run the Streamlit app:

```bash
streamlit run main_smart_assistant_app_ui.py
```

The web interface will launch in your browser at [localhost:8501](http://localhost:8501).

---

## 📝 **Usage Example**

1. Upload your class notes in `.txt`, `.pdf`, or `.docx` format
2. Ask contextual questions like:

   * *“Explain photosynthesis steps.”*
3. Generate quizzes for quick revision:

   * *“Quiz me on this topic.”*
4. Summarize lengthy notes:

   * *“Summarize the chapter on cell biology.”*
5. Detect your emotional sentiment based on inputs.
6. Other than document, even general questions.

---

## 🔧 **System Requirements**

* Python >= 3.8
* Streamlit
* Transformers
* Optimum\[OpenVINO]
* Whisper
* pyttsx3
* Other dependencies listed in `requirements.txt`

---

## 📈 **Future Enhancements**

* Deploy to Hugging Face Spaces or Render
* Integrate multimodal visual question answering
* Support for classroom attendance and analytics
* Personalised student dashboards

---

### ✨ **Acknowledgements**

* Google Generative AI – Gemini APIs
* Hugging Face Transformers & Optimum
* OpenVINO Toolkit
* Streamlit for rapid prototyping

---
