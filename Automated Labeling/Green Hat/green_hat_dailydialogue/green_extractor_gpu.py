#!/usr/bin/env python3
"""
This script:
1. Loads a list of 100 words.
2. Encodes each word into embeddings with facebook/bart-large-mnli, using encoder_last_hidden_state.
3. Runs KMeans clustering (k=5).
4. Picks the single word closest to each cluster centroid.
5. Loads the DailyDialog train split and flattens it into sentences.
6. Uses a zero-shot classification loop with manual per-label batching.
7. Applies the classifier in per-label sub-batches to fit 4GB VRAM.
8. Saves any sentence where the highest label probability ≥ 0.99.
"""
import time
import numpy as np
from sklearn.cluster import KMeans
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
import torch
from tqdm import tqdm

# 1. Your 100 words
words = [
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

# 2. Load tokenizer & model for embeddings
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", use_fast=True)
embed_model = AutoModel.from_pretrained("facebook/bart-large-mnli")
embed_model.eval()
if torch.cuda.is_available():
    embed_model.to("cuda")
embed_model.config.output_hidden_states = True

# 3. Compute embeddings
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    last_hidden = outputs.encoder_last_hidden_state.squeeze(0)
    return last_hidden.mean(dim=0).cpu().numpy()

print("Computing embeddings for 100 words…")
t0 = time.perf_counter()
embeddings = np.vstack([get_embedding(w) for w in words])
print(f"→ Embedding compute time: {time.perf_counter() - t0:.1f}s")

# free embedding model
if torch.cuda.is_available():
    del embed_model
    torch.cuda.empty_cache()

# 4. KMeans clustering
k = 25
print(f"Running KMeans clustering (k={k})…")
t1 = time.perf_counter()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
_ = kmeans.fit_predict(embeddings)
print(f"→ KMeans time: {time.perf_counter() - t1:.1f}s")

# 5. Pick labels
centroids = kmeans.cluster_centers_
cluster_labels = []
for i, center in enumerate(centroids):
    idx = np.argmin(np.linalg.norm(embeddings - center, axis=1))
    cluster_labels.append(words[idx])
    print(f"Cluster {i} label: '{words[idx]}'")

# 6. Load sentences
print("Loading DailyDialog sentences…")
t2 = time.perf_counter()
dd = load_dataset("daily_dialog", split="train")
all_sents = [s for dialog in dd["dialog"] for s in dialog]
print(f"→ Loaded {len(all_sents)} sentences in {time.perf_counter() - t2:.1f}s")

# 7. Classification with per-label batching to reduce memory

def save_high_confidence(sentences, labels, batch_size=64, threshold=0.9):  # lowered default threshold to 0.9
    # Load classification model in FP16
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli", torch_dtype=torch.float16
    )
    cls_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model.to(device)

    entail_idx = cls_model.config.label2id.get("entailment", 2)
    output_file = "high_confidence_sentences.txt"

    with open(output_file, "w", encoding="utf-8") as fout:
        # Process in CPU-side sentence batches
        for start in tqdm(range(0, len(sentences), batch_size), desc="Sentence batches"):
            batch = sentences[start:start+batch_size]
            batch_size_actual = len(batch)
            # Initialize arrays
            max_scores = torch.zeros(batch_size_actual, device=device, dtype=torch.float16)
            best_labels = [-1] * batch_size_actual

            # For each label, do one forward pass
            for lid, lbl in enumerate(labels):
                hypothesis = f"This example is about {lbl}."
                # duplicate hypothesis per sentence
                hypos = [hypothesis] * batch_size_actual
                inputs = tokenizer(
                    batch,
                    hypos,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                # to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        logits = cls_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[:, entail_idx]
                # compare and update
                for i, score in enumerate(probs):
                    if score > max_scores[i]:
                        max_scores[i] = score
                        best_labels[i] = lid
                # free memory fragments
                del inputs, logits, probs
                torch.cuda.empty_cache()

            # Write out high-confidence
            for i, score in enumerate(max_scores):
                if score >= threshold:
                    lid = best_labels[i]
                    fout.write(f"Cluster {lid}\t{labels[lid]}\t{float(score):.4f}\t{batch[i]}\n")
    print(f"Done. Results saved to '{output_file}'")

# 8. Run classification with reduced batch size to fit 4GB VRAM
#    Decrease batch_size to 32 or lower if OOM persists
save_high_confidence(all_sents, cluster_labels, batch_size=32, threshold=0.9)  # lowered threshold to 0.9 for more results


