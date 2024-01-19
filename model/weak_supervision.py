import logging
from typing import List

import torch
from model.model import PairEncoder
from model.util import PairInput
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)


def label_sentences(
    sent_transformer: SentenceTransformer,
    encoder: PairEncoder,
    train: PairInput,
    top_k: int,
    batch_size: int,
    verbose: bool = True,
) -> List[PairInput]:
    """
    Label sentences using weak supervision.

    Args:
        sent_transformer (SentenceTransformer): The sentence transformer model.
        encoder (PairEncoder): The pair encoder model.
        train (PairInput): The training data.
        top_k (int): The number of top similar sentences to consider.
        batch_size (int): The batch size for encoding sentences.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        List[PairInput]: The labeled pair inputs.
    """
    sentences = set()
    for sample in train:
        sentences.update(sample.texts)

    sentences = list(sentences)
    sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)}
    duplicates = set((sent2idx[data.pair[0]], sent2idx[data.pair[1]]) for data in train)

    sentences = list(sentences)
    if "e5" in sent_transformer.name:
        logger.info("Adding 'query:' prefix to sentences")
        query_sents = [f"query: {s}" for s in sentences]

    embeddings = sent_transformer.encode(
        query_sents,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=verbose,
    )
    weak_data = []
    logging.info(f"Performing weak supervision with {sent_transformer}...")
    for idx in range(len(sentences)):
        sentence_embedding = embeddings[idx]
        cos_scores = cos_sim(sentence_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k + 1)

        for _, _id in zip(top_results[0], top_results[1]):
            if _id != idx and (_id, idx) not in duplicates:
                weak_data.append((sentences[idx], sentences[_id]))
                duplicates.add((idx, _id))

    weak_scores = encoder.predict(weak_data)
    assert all(0.0 <= score <= 1.0 for score in weak_scores)

    return list(
        PairInput(pair=(data[0], data[1]), label=score)
        for data, score in zip(weak_data, weak_scores)
    )
