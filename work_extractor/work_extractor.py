import argparse
import time
from transformers import pipeline
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Extract 100 work-related sentences from DailyDialog.")
    parser.add_argument('--output', '-o', type=str, default='work_sentences.txt',
                        help='Output file (default: work_sentences.txt)')
    return parser.parse_args()

def load_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli",
        device=0  # Use GPU if available, else -1 for CPU
    )

def main():
    args = parse_args()

    print("🔄 Loading DailyDialog dataset...")
    dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)

    print("🧠 Loading zero-shot model...")
    classifier = load_model()

    work_sentences = []
    total = len(dataset)
    start_time = time.time()

    for idx, dialogue in enumerate(dataset):
        for sentence in dialogue['dialog']:
            try:
                result = classifier(sentence, candidate_labels=["work", "not work"])
                score_work = result['scores'][result['labels'].index("work")]
                score_not_work = result['scores'][result['labels'].index("not work")]

                if score_work >= 0.01 and score_not_work < 0.01:
                    work_sentences.append(sentence)
                    print(f"✅ [{len(work_sentences)}] {sentence} (work: {score_work:.3f})")

                    if len(work_sentences) == 100:
                        break
            except Exception as e:
                print(f"❌ Error classifying sentence: {e}")

        if len(work_sentences) >= 100:
            break

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"{idx+1}/{total} dialogues processed... ⏱️ Elapsed: {elapsed:.1f}s")

    # Save results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(work_sentences, 1):
                f.write(f"{i}. {sentence}\n\n")
        print(f"\n📁 Saved {len(work_sentences)} work-related sentences to '{args.output}'")
    except Exception as e:
        print(f"❌ Error saving output: {e}")

if __name__ == '__main__':
    main()