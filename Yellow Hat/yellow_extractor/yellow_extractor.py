import argparse
import time
from transformers import pipeline
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Extract creative sentences from the DailyDialog dataset.")
    parser.add_argument('--output', '-o', type=str, default='creative_dialogues.txt',
                        help='Output file in human-readable format (default: creative_dialogues.txt)')
    parser.add_argument('--min_prob', type=float, default=0.97,
                        help='Minimum probability to consider a sentence creative (default: 0.97)')
    return parser.parse_args()

def load_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli",
        framework="pt",
        device=0  # use GPU if available
    )

def main():
    args = parse_args()

    print("Loading DailyDialog dataset...")
    dataset = load_dataset("daily_dialog", split="train")

    print("Loading zero-shot model, please wait...")
    classifier = load_model()

    creative_sentences = []
    total = len(dataset)
    start_time = time.time()

    for idx, dialogue in enumerate(dataset):
        for sentence in dialogue['dialog']:
            try:
                result = classifier(sentence, candidate_labels=['creative']) # but gets yellow hat sentences...
                score = result['scores'][0]
                if score >= args.min_prob:
                    creative_sentences.append((score, sentence))
            except Exception as e:
                print(f"Error classifying sentence: {e}")

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            remaining = (elapsed / (idx + 1)) * (total - idx - 1)
            print(f"{idx+1}/{total} dialogues processed - Estimated time remaining: {remaining:.1f} seconds")

    # Save results in human-readable format
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, (prob, sent) in enumerate(creative_sentences, 1):
                f.write(f"{i}. Probability: {prob:.4f}\n{sent}\n\n")
        print(f"\n✅ Saved {len(creative_sentences)} creative sentences to '{args.output}' (threshold ≥ {args.min_prob})")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    main()
