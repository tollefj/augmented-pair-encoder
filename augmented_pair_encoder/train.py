import logging
import os
from datetime import datetime
from math import ceil
from typing import List

from fire import Fire
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from augmented_pair_encoder.evaluation import CorrelationEvaluator
from augmented_pair_encoder.model import PairEncoder
from augmented_pair_encoder.util import PairInput
from augmented_pair_encoder.weak_supervision import label_sentences


def train_encoder(
    train_samples: List[PairInput],
    evaluator: CorrelationEvaluator = None,
    model_name: str = "bert-base-uncased",
    similarity_model: str = None,
    batch_size: int = 32,
    learning_rate=8e-5,
    epochs=10,
    eval_steps=0,  # if 0, evaluate after each epoch
    max_length=128,
    k=0,
    weak_training_epochs=1,  # if k>0, train a weak model for this many epochs before weak supervision
    seed: int = None,
    save_to=None,
    verbose=False,
    device="cuda",
    logfile=None,
    timestamp=None,
):
    encoder = PairEncoder(model_name, device=device, max_length=max_length, seed=seed)
    starttime = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=logfile,
    )
    logging_param = {
        "PairEncoder": model_name,
        "batch": batch_size,
        "lr": learning_rate,
        "epochs": epochs,
        "k": k,
    }
    logging.info(logging_param)

    if k > 0 and similarity_model:
        logging.info("Training encoder for 1 epoch prior to weak supervision...")
        dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
        encoder.fit(
            dataloader=dataloader,
            evaluator=evaluator,
            epochs=weak_training_epochs,
            warmup_steps=ceil(len(dataloader) * 1 * 0.1),
            learning_rate=learning_rate,
            verbose=verbose,
        )
        logging.info("Weakly labeling sentences...")
        sent_transformer = SentenceTransformer(similarity_model, device=device)
        weak_samples = label_sentences(
            sent_transformer=sent_transformer,
            encoder=encoder,
            train=train_samples,
            top_k=k,
            batch_size=batch_size,
        )
        train_samples += weak_samples

    output_path = None
    if save_to:
        model_id = model_name.replace("/", "-")
        output_path = os.path.join(save_to, f"{model_id}-{starttime}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    encoder.fit(
        dataloader=dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=ceil(len(dataloader) * epochs * 0.1),
        learning_rate=learning_rate,
        verbose=verbose,
        output_path=output_path,
        evaluation_steps=eval_steps,
    )

    return encoder


if __name__ == "__main__":
    Fire(train_encoder)
