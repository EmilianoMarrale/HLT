#!/usr/bin/env python3
"""
This script:
1. Loads a list of 100 words.
2. Encodes each word into embeddings with facebook/bart-large-mnli, using encoder_last_hidden_state.
3. Uses each word as a label for zero-shot classification.
4. Loads the DailyDialog train split and flattens it into sentences.
5. Uses a zero-shot classification loop with manual per-label batching.
6. Applies the classifier in per-label sub-batches to fit 4GB VRAM.
7. Saves any sentence where the highest label probability ≥ threshold.
"""
import time
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
import torch
from tqdm import tqdm

# 1. Your 100 words (labels)
labels = [
    "creative", "creativity", "cognitive", "domain", "innovation", "openness",
    "because", "divergent", "process", "motivation", "domains", "found",
    "abilities", "thinking", "scores", "solving", "individuals", "personality",
    "scales", "processes", "empirical", "ratings", "correlations", "originality",
    "traits", "associative", "influences", "primary", "conceptual", "instance",
    "developmental", "individual", "problem", "intrinsic", "artistic",
    "evolutionary", "correlated", "ability", "programs", "intelligence",
    "cannot", "facilitate", "toward", "correlation", "basis", "computational",
    "extrinsic", "selective", "cognition", "hypothesis", "interactions",
    "criterion", "validity", "according", "measures", "tests", "verbal",
    "investigations", "heuristics", "fluency", "rated", "psychologists",
    "complexity", "discoveries", "semantic", "discovery", "schema", "rat",
    "unconscious", "probability", "self", "knowledge", "variables", "primitive",
    "novelty", "subjects", "retention", "dimensions", "hypotheses", "innovative",
    "ideas", "related", "dimension", "validation", "attributes", "research",
    "iq", "artefacts", "combinations", "predictions", "heuristic", "factors",
    "these", "psychology", "barren", "positively", "investigators",
    "perceptual", "example", "elements"
]

# 2. Load tokenizer & embedding model (for embeddings if needed)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", use_fast=True)
embed_model = AutoModel.from_pretrained("facebook/bart-large-mnli")
embed_model.eval()
if torch.cuda.is_available():
    embed_model.to("cuda")
embed_model.config.output_hidden_states = True

# Function to get embedding (unused now, but kept if needed for advanced tasks)
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    last_hidden = outputs.encoder_last_hidden_state.squeeze(0)
    return last_hidden.mean(dim=0).cpu().numpy()

# Free embedding model if not used further
after_embeddings = True
if after_embeddings and torch.cuda.is_available():
    del embed_model
    torch.cuda.empty_cache()

# 3. Load sentences from DailyDialog
print("Loading DailyDialog sentences…")
t0 = time.perf_counter()
dd = load_dataset("daily_dialog", split="train")
all_sents = [s for dialog in dd["dialog"] for s in dialog]
print(f"→ Loaded {len(all_sents)} sentences in {time.perf_counter() - t0:.1f}s")

# 4. Zero-shot classification with per-label batching to reduce memory

def save_high_confidence(sentences, labels, batch_size=64, threshold=0.9):
    # Load classification model in FP16 for efficiency
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli", torch_dtype=torch.float16
    )
    cls_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model.to(device)

    entail_idx = cls_model.config.label2id.get("entailment", 2)
    output_file = "high_confidence_sentences.txt"

    with open(output_file, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(sentences), batch_size), desc="Sentence batches"):
            batch = sentences[start:start+batch_size]
            actual_size = len(batch)
            max_scores = torch.zeros(actual_size, device=device, dtype=torch.float16)
            best_labels = [-1] * actual_size

            # For each label, do one forward pass against the batch
            for lid, label in enumerate(labels):
                hypothesis = f"This example is about {label}."
                hypos = [hypothesis] * actual_size
                inputs = tokenizer(
                    batch,
                    hypos,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        logits = cls_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[:, entail_idx]

                # Update best scores and labels per sentence
                for i, score in enumerate(probs):
                    if score > max_scores[i]:
                        max_scores[i] = score
                        best_labels[i] = lid

                del inputs, logits, probs
                torch.cuda.empty_cache()

            # Write out high-confidence results
            for i, score in enumerate(max_scores):
                if score >= threshold:
                    lid = best_labels[i]
                    fout.write(f"Label {lid}\t{labels[lid]}\t{float(score):.4f}\t{batch[i]}\n")
    print(f"Done. Results saved to '{output_file}'")

# 5. Run classification
#    Adjust batch_size and threshold as needed
save_high_confidence(all_sents, labels, batch_size=64, threshold=0.9)
