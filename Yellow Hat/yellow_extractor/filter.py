import re
from transformers import pipeline

# 1. Carica modello zero-shot
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    tokenizer="facebook/bart-large-mnli",
    device=0  # metti -1 se vuoi forzare CPU
)

# 2. Parametri
INPUT_FILE = "creative_dialogues.txt"
OUTPUT_FILE = "filtered_creative_sentences.txt"
MAX_NON_CREATIVE_PROB = 0.01

# 3. Funzione per estrarre le frasi dal file
def extract_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Estrae blocchi di testo separati da due newline
    blocks = re.split(r"\n\s*\n", text.strip())
    sentences = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            sentence = lines[1].strip()
            sentences.append(sentence)
    return sentences

# 4. Elabora frasi e filtra
def filter_creative(sentences):
    valid = []
    for i, sent in enumerate(sentences, 1):
        try:
            result = classifier(
                sent,
                candidate_labels=["creative", "not creative"]
            )
            # Trova punteggio "not creative"
            idx = result["labels"].index("not creative")
            not_creative_score = result["scores"][idx]

            if not_creative_score < MAX_NON_CREATIVE_PROB:
                valid.append((sent, 1.0 - not_creative_score))
                print(f"✅ Kept [{i}]: not_creative = {not_creative_score:.4f}")
            else:
                print(f"❌ Skipped [{i}]: not_creative = {not_creative_score:.4f}")
        except Exception as e:
            print(f"⚠️ Error at [{i}]: {e}")
    return valid

# 5. Salva su file
def save_output(filtered, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (sentence, score) in enumerate(filtered, 1):
            f.write(f"{i}. Estimated Creative Probability: {score:.4f}\n{sentence}\n\n")
    print(f"\n💾 Saved {len(filtered)} creative-like sentences to: {path}")

# === MAIN ===
if __name__ == "__main__":
    all_sentences = extract_sentences(INPUT_FILE)
    filtered_sentences = filter_creative(all_sentences)
    save_output(filtered_sentences, OUTPUT_FILE)
