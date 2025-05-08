import json
from nltk.corpus import opinion_lexicon # For optimism YELLOW HAT
from empath import Empath # For emotions RED HAT
from nltk.corpus import wordnet# Creative/Problem Solving GREEN HAT
import nltk

json_data = {}
with open("dailydialog/attitude_emotion_dialogues.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

empath_categories = ["red_hat_rage", "red_hat_love", "red_hat_disappointment",  "yellow_hat"]
emotion_words = Empath()
empath_yellow_hat = ["optimism", "hope", "trust", "confidence"]

empath_red_hat_rage = ["anger", "hate", "aggression", "rage", "envy"]
empath_red_hat_love = ["love", "joy", "trust", "confidence", "sympathy", "pride"]
empath_red_hat_disappointment = ["disappointment", "shame", "sadness", "fear", "nervousness", "disgust"]
#empath_red_hat = ["irritability" "exasperation", "fear", "nervousness", "disgust", "sadness", "emotional"]
emotion_words.create_category("yellow_hat", empath_yellow_hat)
emotion_words.create_category("red_hat_rage", empath_red_hat_rage)
emotion_words.create_category("red_hat_love", empath_red_hat_love)
emotion_words.create_category("red_hat_disappointment", empath_red_hat_disappointment)


for dialogue in json_data[0:10]:
    for utterance in dialogue["utterances"]:
        print(utterance["utterance"])
        res = emotion_words.analyze(utterance["utterance"], categories=empath_categories, normalize=True)
        print(res)
        print("\n")

empath_white_hat = {""} # If


empath_black_hat = {"communication"}


empath_green_hat = {"creativity", "problem solving", "innovation", "design", "create"}