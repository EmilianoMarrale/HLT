from enum import Enum
import json

class Topic(Enum):
    Ordinary_Life = "1"
    School_Life = "2"
    Culture_Education = "3"
    Attitude_Emotion = "4"
    Relationship = "5"
    Tourism = "6"
    Health = "7"
    Work = "8"
    Politics = "9"
    Finance = "10"

class Emotion(Enum):
    no_emotion = "0"
    anger = "1"
    disgust = "2"
    fear = "3"
    happiness = "4"
    sadness = "5"
    surprise = "6"

# Usage example Act(2).name = question
class Act(Enum):
    inform = "1"
    question = "2"
    directive = "3"
    commissive = "4"


class Utterance:
    def __init__(self, turn, utterance, emotion, act, hat=""):
        self.turn = turn
        self.utterance = utterance
        self.emotion = emotion
        self.act = act
        self.hat = hat

    def __str__(self):
        return (f"Turn {self.turn}: \n\t"
                + f"Utterance: {self.utterance} \n\t"
                + f"Emotion: {self.emotion} \n\t"
                + f"Act: {self.act} \n\t"
                + f"HAT: {self.hat} \n")

    def to_dict(self):
        return {
            "turn": self.turn,
            "utterance": self.utterance,
            "emotion": self.emotion,
            "act": self.act,
            "hat": self.hat
        }

class Dialogue:
    def __init__(self, id: int, utterances: list, topic: str):
        self.id = id
        self.utterances = utterances
        self.topic = topic

    def __str__(self):
        return (f"Dialogue {self.id}: \n\t"
                + f"Topic: {self.topic} \n\t"
                + f"Utterances: \n\t"
                + "\n\t".join([str(u) for u in self.utterances]) + "\n")

    def to_dict(self):
        return {
            "id": self.id,
            "topic": self.topic,
            "utterances": [u.__dict__ for u in self.utterances]
        }

acts = open("dailydialog/dialogues_act.txt", "r", encoding="utf-8").readlines()
emotions = open("dailydialog/dialogues_emotion.txt", "r", encoding="utf-8").readlines()
topics = open("dailydialog/dialogues_topic.txt", "r", encoding="utf-8").readlines()

EOU = " __eou__ "
dialogues = []
with open("dailydialog/dialogues_text.txt", "r", encoding="utf-8") as f:
    dialogue_id = 0
    for dialogue in f:
        # print(dialogue)

        utterances_txt = dialogue.strip().split(EOU)
        utterances_obj = []
        for turn, u in enumerate(utterances_txt):
            acts_array = acts[dialogue_id].strip().split(" ")
            emotions_array = emotions[dialogue_id].strip().split(" ")
            if len(utterances_txt) != len(acts_array) or len(utterances_txt) != len(emotions_array):
                print(
                    f"Skipping dialogue {dialogue_id}: utterances={len(utterances_txt)}, acts={len(acts_array)}, emotions={len(emotions_array)}")
                continue
            # print("emotions length: " + str(len(emotions_array)) + " turn: " + str(turn) + " id: " + str(dialogue_id))
            # print(Emotion(emotions_array[turn]).name)
            emotion = Emotion(emotions_array[turn]).name
            act = Act(acts_array[turn]).name
            utterances_obj.append(Utterance(turn, u, emotion, act))

        topic = Topic(topics[dialogue_id].strip()).name
        dialogue = Dialogue(dialogue_id, utterances_obj, topic)
        dialogues.append(dialogue)
        if dialogue_id == 100:
            break
        dialogue_id += 1


# Save the dialogues to a JSON file
with open("dailydialog/dialogues_100.json", "w", encoding="utf-8") as f:
    json.dump([dialogue.to_dict() for dialogue in dialogues], f, ensure_ascii=False, indent=4)

json_data = {}
with open("dailydialog/dialogues.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)


work_dialogues = []
relationship_dialogues = []
attitude_emotion_dialogues = []
for dialogue in json_data:
    if dialogue["topic"] == "Work":
        work_dialogues.append(dialogue)
    if dialogue["topic"] == "Relationship":
        relationship_dialogues.append(dialogue)
    if dialogue["topic"] == "Attitude_Emotion":
        attitude_emotion_dialogues.append(dialogue)

with open("dailydialog/work_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(work_dialogues, f, ensure_ascii=False, indent=4)

with open("dailydialog/relationship_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(relationship_dialogues, f, ensure_ascii=False, indent=4)

with open("dailydialog/attitude_emotion_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(attitude_emotion_dialogues, f, ensure_ascii=False, indent=4)