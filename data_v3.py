"""v3 data mix: synth (in-domain skill) + public (zero-shot generality) + instruction diversity.

Root cause of v2's collapse (AG 0.325, flip 0.53): fixed instruction phrasing +
fixed option sets -> template matching, no semantic decision ability.
Fix: 6-8 instruction paraphrases per type, varied criteria wording, ALWAYS shuffle
options, public classification data rendered as choice/noul questions.
"""
import random
import numpy as np

CHOICE_INS = [
    "Which category best describes this text?",
    "Classify the following text into exactly one option.",
    "Which label applies to this input?",
    "Route this item to the correct category.",
    "Select the single best-fitting option for the text below.",
    "Which topic does this text belong to?",
    "Assign the most appropriate label.",
    "Pick the option that fits this input best.",
]
SCORE_INS = [
    "How urgent is this?",
    "Rate the urgency of this request.",
    "On the scale below, where does this fall?",
]
NOUL_INS = [
    "Does the statement hold for this text?",
    "Is the following true of the input?",
    "Answer yes or no for the input below.",
]

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
AG_DESCS = [
    {"World": "international news, politics, world events", "Sports": "sports, games, athletes",
     "Business": "business, markets, companies", "Sci/Tech": "science, technology"},
    {"World": "world", "Sports": "sports", "Business": "business", "Sci/Tech": "sci/tech"},
    {"World": "global affairs and politics", "Sports": "athletics and competition",
     "Business": "economy and corporate news", "Sci/Tech": "research, computing, space"},
]
EM_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EM_DESCS = [
    {l: l for l in EM_LABELS},
    {"sadness": "sad, grief, disappointment", "joy": "happy, delighted", "love": "affectionate, loving",
     "anger": "angry, furious", "fear": "afraid, anxious", "surprise": "surprised, amazed"},
]


def public_records(n_ag=2200, n_em=1200, seed=0):
    from datasets import load_dataset
    rng = random.Random(seed)
    recs = []
    ag = load_dataset("fancyzhx/ag_news", split=f"train[:{n_ag}]")
    for r in ag:
        y = int(r["label"])
        recs.append({"kind": "q", "wf": "general",
                     "state": r["text"][:1500],
                     "q": {"t": "choice", "ins": rng.choice(CHOICE_INS),
                           "crit": rng.choice(AG_DESCS)},
                     "soft": [1.0 if i == y else 0.0 for i in range(4)], "y": y})
    em = load_dataset("dair-ai/emotion", split=f"train[:{n_em}]")
    for r in em:
        y = int(r["label"])
        if rng.random() < 0.35:
            # binarized noul: is this joy/love (positive)?
            pos = 1 if y in (1, 2) else 0
            p1 = 0.92 if pos else 0.08
            recs.append({"kind": "q", "wf": "general", "state": r["text"][:800],
                         "q": {"t": "noul", "ins": "Is the sentiment of this text positive?",
                               "crit": {}},
                         "soft": [1-p1, p1], "y": pos})
        else:
            recs.append({"kind": "q", "wf": "general", "state": r["text"][:800],
                         "q": {"t": "choice", "ins": rng.choice(CHOICE_INS),
                               "crit": rng.choice(EM_DESCS)},
                         "soft": [1.0 if i == y else 0.0 for i in range(6)], "y": y})
    rng.shuffle(recs)
    return recs
