# 🐍 Convomate Python NLP Module

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/Flask-3.0.3-brightgreen.svg)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)

Welcome to the **Convomate Python NLP Module**. This repository hosts a suite of Python Flask-based microservices that leverage machine learning and deep learning models to perform complex natural language processing (NLP) tasks for the Convomate ecosystem.

---

## ⚡ NLP Microservices Overview

The repository is structured as a collection of independent, dockerized microservices, each running a Flask API:

### 1. 🎙️ Grammar Practice Chatbot (`/Chatbot`)
- **Engine:** `vennify/t5-base-grammar-correction` (T5 sequence-to-sequence model)
- **Role:** An interactive English learning partner. It conducts chats by prompting starter questions, detects topics dynamically (food, music, movies, travel, etc.), evaluates user responses for grammatical errors, returns helpful corrections with friendly compliments, and guides the conversation with follow-ups.
- **Port:** `5003`

### 2. 📝 Advanced Paragraph Corrector (`/Paragraph_Checker`)
- **Pipeline:** Combining rule-based logic and deep learning:
  1. `language-tool-python` (Wrapper for LanguageTool en-US) for spelling, capitalization, and simple grammar checks.
  2. `vennify/t5-base-grammar-correction` for neural-based syntax repairs.
  3. `humarin/chatgpt_paraphraser_on_T5_base` for tense, phrasing, and style refinements.
- **Port:** `5001`

### 3. 🎯 Summarizer (`/summarizer`)
- **Engine:** Natural Language Toolkit (`nltk` sent_tokenize) and sentence scoring based on word-frequency heuristics. Supports output as unified paragraphs or clean bullet points.
- **Port:** `5002`

### 4. 🌐 Translator (`/Translator`)
- **Engine:** `googletrans` API wrapper.
- **Role:** Implements language detection and bidirectional translation, including phonetic transcriptions/pronunciation guides for foreign characters.
- **Port:** `5000`

---

## 📂 Project Directory Structure

```bash
Convomate-Python-module/
├── Chatbot/                            # Conversational Practice Bot
│   ├── app.py                          # Flask entrypoint (Port 5003)
│   ├── grammar_chatbot.py              # Conversational logic & topic parsing
│   └── requirements.txt
├── Paragraph_Checker/                  # Multi-layer Grammar Optimizer
│   ├── app.py                          # Flask entrypoint (Port 5001)
│   ├── paragraph_checker.py            # Combined ML & rule-based pipeline
│   └── requirements.txt
├── summarizer/                         # Text Summarizer
│   ├── app.py                          # Flask entrypoint (Port 5002)
│   ├── summarizer_logic.py             # NLTK-based scoring logic
│   └── requirements.txt
├── Translator/                         # Language Translation
│   ├── app.py                          # Flask entrypoint (Port 5000)
│   ├── translator_logic.py             # googletrans integration
│   └── requirements.txt
├── docker-compose-image-create.yml     # Compiles Docker images for services
└── docker-compose-run.yml              # Spins up the microservice fleet
```

---

## 🚀 Installation & Local Setup

### Prerequisites
- **Python 3.8+**
- **pip** and **virtualenv**
- (Optional) **Java Development Kit (JDK)** - Required by LanguageTool's underlying local server.

### Running a Service Individually
For example, to run the **Paragraph Checker**:
```bash
cd Paragraph_Checker
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

---

## 🐳 Docker Orchestration

You can spin up the entire suite of services using Docker.

### 1. Build Docker Images
```bash
docker compose -f docker-compose-image-create.yml build
```

### 2. Run Services Fleet
```bash
docker compose -f docker-compose-run.yml up
```
This maps the following endpoints:
- `http://localhost:5000/translate` (Translator)
- `http://localhost:5001/correct_text` (Grammar Corrector)
- `http://localhost:5002/summarize` (Summarizer)
- `http://localhost:5003/chat` & `/start` (Chatbot)

---

## 👥 Related Repositories

- ☕ **[CONVOMATE-BACKEND](https://github.com/Shevadesuyash/CONVOMATE-BACKEND)** - Spring Boot routing & authentication core.
- 🎨 **[CONVOMATE-FRONTEND](https://github.com/Shevadesuyash/CONVOMATE-FRONTEND)** - React-based user interface.
