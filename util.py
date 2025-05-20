import json
import pandas as pd

def get_dataframe_from_json(json_file):
    with open(json_file, 'r') as f:
        dialogues = json.load(f)

    utterance_df = pd.DataFrame()
    for i in range(1, len(dialogues)):
        utterance_df = pd.concat([utterance_df, pd.DataFrame(dialogues[i]["utterances"])], ignore_index=True)
    return utterance_df