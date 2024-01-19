import numpy as np
from augmented_pair_encoder.util import PairInput

sources = {
    "stsb": "mteb/stsbenchmark-sts",
    "sickr": "mteb/sickr-sts",
}


def normalize(scores):
    if isinstance(scores, list):
        scores = np.array(scores)
    scores = scores / 5.0
    scores = np.round(scores, 3)
    return scores


def create_pair(row):
    return PairInput(pair=(row["sentence1"], row["sentence2"]), label=row["score"])


def process_sts(dataset, _drop_duplicates=True, _normalize=True, return_pairs=True):
    df = dataset.to_pandas()
    df = df[["score", "sentence1", "sentence2"]]
    if _normalize:
        # normalize scores
        df.score = normalize(df.score)
    # remove duplicate sentence1, sentence2 pairs
    if _drop_duplicates:
        df = df.drop_duplicates(subset=["sentence1", "sentence2"])

    if return_pairs:
        df = df.apply(create_pair, axis=1).tolist()
    return df
