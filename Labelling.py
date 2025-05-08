import json
from nltk.corpus import opinion_lexicon # For optimism YELLOW HAT
from empath import Empath # For emotions RED HAT
from nltk.corpus import wordnet# Creative/Problem Solving GREEN HAT
import nltk

json_data = {}
with open("dailydialog/dialogues.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

empath_categories = ["red_hat", "yellow_hat"]
emotion_words = Empath()
empath_yellow_hat = ["optimism", "hope", "trust", "confidence"]
empath_red_hat = ["disappointment", "anger", "shame", "joy", "love", "exasperation", "irritability", "fear", "nervousness", "pride", "hate", "aggression", "envy", "sympathy", "disgust", "rage", "sadness", "emotional"]

emotion_words.create_category("red_hat", empath_red_hat)
emotion_words.create_category("yellow_hat", empath_yellow_hat)

for dialogue in json_data[0:10]:
    for utterance in dialogue["utterances"]:
        print(utterance["utterance"])
        res = emotion_words.analyze(utterance["utterance"], categories=empath_categories, normalize=True)
        print("Risultato empath hat per  " + str(utterance["utterance"]))
        print(res)

empath_white_hat = {""} # If


empath_black_hat = {"communication"}


empath_green_hat = {"creativity", "problem solving", "innovation", "design", "create"}
