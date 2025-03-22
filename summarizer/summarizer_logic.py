#summarizer
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]
