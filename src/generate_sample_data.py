"""Generate a small synthetic review dataset so the project runs without downloads.

Replace data/sample_reviews.csv with a real dataset (Amazon / Yelp from Kaggle) for real results.
"""
import random

import pandas as pd

random.seed(42)

ASPECTS = ["battery life", "screen", "camera", "delivery", "customer service", "price", "build quality",
           "sound quality", "packaging", "software", "charger", "design"]

POS = ["The {a} is excellent.", "I love the {a}.", "Really impressed with the {a}.",
       "The {a} is amazing and works perfectly.", "Great {a}, highly recommend.", "The {a} exceeded my expectations."]
NEG = ["The {a} is terrible.", "Very disappointed with the {a}.", "The {a} stopped working after a week.",
       "Worst {a} I have ever seen.", "The {a} is poor and overpriced.", "I hate the {a}."]
NEU = ["The {a} is okay.", "The {a} is average, nothing special.", "The {a} is fine but not great.",
       "The {a} does the job, I guess.", "Decent {a}, nothing more."]
OPEN_POS = ["Happy with this purchase.", "Overall a great buy.", "Would buy again."]
OPEN_NEG = ["Do not buy this.", "Total waste of money.", "Overall a bad experience."]
OPEN_NEU = ["It is an average product.", "Mixed feelings overall.", "Not bad, not great."]


def make_review():
    r = random.random()
    if r < 0.40:
        label, rating = "positive", random.choice([4, 5])
    elif r < 0.75:
        label, rating = "negative", random.choice([1, 2])
    else:
        label, rating = "neutral", 3
    pool = {"positive": (POS, OPEN_POS, NEG), "negative": (NEG, OPEN_NEG, POS), "neutral": (NEU, OPEN_NEU, POS)}[label]
    main, opener, contrast = pool
    aspects = random.sample(ASPECTS, random.randint(2, 3))
    sents = [random.choice(main).format(a=aspects[0]), random.choice(main).format(a=aspects[1])]
    # occasional mixed review: one sentence about a different aspect with opposite tone
    if len(aspects) == 3 and label != "neutral" and random.random() < 0.5:
        sents.append(random.choice(contrast).format(a=aspects[2]))
    sents.append(random.choice(opener))
    random.shuffle(sents)
    return {"review": " ".join(sents), "rating": rating}


if __name__ == "__main__":
    df = pd.DataFrame([make_review() for _ in range(1500)])
    df.to_csv("data/sample_reviews.csv", index=False)
    print(df["rating"].value_counts().sort_index())
    print("Saved data/sample_reviews.csv with", len(df), "reviews")
