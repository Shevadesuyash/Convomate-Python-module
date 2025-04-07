#summarizer
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

nltk.download('punkt_tab')

def summarize_text(text, format_type="paragraph"):
    sentences = sent_tokenize(text)
    word_counts = Counter(text.split())
    
    # Score sentences based on word frequency
    sentence_scores = {sentence: sum(word_counts[word] for word in sentence.split()) for sentence in sentences}
    
    # Select top sentences (30% of total sentences)
    num_sentences = max(1, len(sentences) // 3)
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    if format_type == "points":
        return "\n".join(f"- {sentence.strip()}" for sentence in summary_sentences)
        
    return " ".join(summary_sentences)
par = input()
print()
typ_for = input('enter type of summarize u want , write \'points\'  or \' paragraph\' ')
summarize_text(par , typ_for)