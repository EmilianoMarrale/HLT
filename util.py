import json
import pandas as pd
from eda import eda

def get_dataframe_from_json(json_file):
    with open(json_file, 'r') as f:
        dialogues = json.load(f)

    utterance_df = pd.DataFrame()
    for i in range(1, len(dialogues)):
        utterance_df = pd.concat([utterance_df, pd.DataFrame(dialogues[i]["utterances"])], ignore_index=True)
    return utterance_df


def eda_augment_dataset(dataframe, num_aug=4, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
    """
    Perform EDA augmentation on the given dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing the text data.
        num_aug (int): Number of augmentations to perform.
        alpha_sr (float): Probability of synonym replacement.
        alpha_ri (float): Probability of random insertion.
        alpha_rs (float): Probability of random swap.
        p_rd (float): Probability of random deletion.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """

    augmented_data = []
    for _, row in dataframe.iterrows():
        text = row['utterance']
        augmented_texts = eda(text, num_aug=num_aug, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd)
        for aug_text in augmented_texts:
            augmented_data.append({'turn': row['turn'], 'utterance': aug_text, 'emotion': row['emotion'], 'act': row['act'], 'hat': row['hat']})

    return pd.DataFrame(augmented_data)
