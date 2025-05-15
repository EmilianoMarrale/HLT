import argparse
import os
import csv

"""
Script to convert a TSV file with 'sentence' and 'probability' columns into a human-readable TXT dataset.
Each entry is numbered, shows probability, and the sentence on the next line.
Filters out rows with probability < 0.95.
Usage:
    python pretty_dataset.py --input fout.txt --output dataset.txt --min_prob 0.95
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Create a readable TXT dataset from a TSV file with filtering.")
    parser.add_argument('--input', '-i', type=str, default='fout.txt',
                        help='Path to the input TSV file (default: out.txt)')
    parser.add_argument('--output', '-o', type=str, default='dataset.txt',
                        help='Path for the output TXT file (default: dataset.txt)')
    parser.add_argument('--min_prob', type=float, default=0.95,
                        help='Minimum probability threshold to include a row (default: 0.95)')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output
    min_prob = args.min_prob

    if not os.path.isfile(input_path):
        print(f"Error: input file '{input_path}' not found.")
        return

    try:
        with open(input_path, 'r', encoding='utf-8', newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', quotechar='"')
            # Ensure required columns
            if not reader.fieldnames or 'sentence' not in reader.fieldnames or 'probability' not in reader.fieldnames:
                print(f"Error: input file must contain 'sentence' and 'probability' columns.")
                return

            with open(output_path, 'w', encoding='utf-8') as txtfile:
                entry_count = 0
                for row in reader:
                    prob_str = row.get('probability', '').strip()
                    try:
                        prob_val = float(prob_str)
                    except ValueError:
                        continue  # skip rows with non-numeric probabilities
                    # Filter out rows below threshold
                    if prob_val < min_prob:
                        continue
                    sent = row.get('sentence', '').strip()
                    entry_count += 1
                    # Write entry
                    txtfile.write(f"{entry_count}. Probability: {prob_val:.4f}\n")
                    txtfile.write(f"{sent}\n\n")
        print(f"Successfully created readable dataset '{output_path}' with {entry_count} entries (min_prob={min_prob}).")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == '__main__':
    main()
