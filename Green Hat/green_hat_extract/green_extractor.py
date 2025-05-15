import os
import re
import sys
import pandas as pd
import argparse
import pdfplumber
import time
import logging
from transformers import pipeline

# Suppress PDF warnings and logs
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Script that extracts sentences with a creativity probability >= threshold
# Processes .txt and .pdf files in a local Windows directory.
# Reports estimated remaining time during execution.

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract creative sentences from .txt and .pdf files in a directory with ETA.')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Directory containing .txt and .pdf files (default: current directory)')
    parser.add_argument('--output', type=str, default='out.txt',
                        help='Output TSV file (default: out.txt)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Minimum probability threshold to consider a sentence creative (default: 0.7)')
    return parser.parse_args()


def extract_text(file_path):
    """Read text from .txt or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ''
    elif ext == '.pdf':
        text = ''
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        except Exception as e:
            print(f"Error extracting PDF {file_path}: {e}")
        return text
    else:
        return ''


def tokenize_sentences(text):
    """Split text into sentences using regex."""
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def load_zero_shot_model():
    """Load zero-shot classification pipeline using MNLI model (PyTorch backend)."""
    # Use keyword arguments for model and tokenizer to ensure correct inference
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli",
        framework="pt",
        device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1
    )


def compute_zero_shot_score(sentence, classifier):
    """Compute the probability that a sentence is creative."""
    labels = ['creative']
    try:
        result = classifier(sentence, labels)
        # result is dict with 'scores' list
        return result['scores'][0]
    except Exception as e:
        print(f"Error classifying sentence: {e}")
        return 0.0


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output
    threshold = args.threshold

    # Gather files and count total sentences for ETA
    print("Scanning files to estimate workload...")
    file_paths = []
    total_sentences = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(('.txt', '.pdf')):
                continue
            path = os.path.join(root, fname)
            text = extract_text(path)
            if text.strip():
                sents = tokenize_sentences(text)
                count = len(sents)
                if count > 0:
                    file_paths.append(path)
                    total_sentences += count
    print(f"Found {len(file_paths)} files with a total of {total_sentences} sentences.")

    # Load zero-shot model
    start_time = time.time()
    print("Loading zero-shot model, please wait...")
    classifier = load_zero_shot_model()

    processed = 0
    results = []
    # Process each file and sentence
    for path in file_paths:
        relative_path = os.path.relpath(path, input_dir)
        text = extract_text(path)
        sents = tokenize_sentences(text)
        for sent in sents:
            if not sent:
                continue
            processed += 1
            score = compute_zero_shot_score(sent, classifier)
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total_sentences - processed)
            print(f"Processed {processed}/{total_sentences} sentences. Estimated time remaining: {remaining:.1f} seconds.")
            if score >= threshold:
                results.append({'file': relative_path, 'sentence': sent, 'probability': score})

    # Save results to TSV
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
        print(f"Processed {len(results)} creative sentences. Results saved to '{output_path}'.")
    else:
        print(f"No creative sentences found with threshold >= {threshold}.")

if __name__ == '__main__':
    main()
