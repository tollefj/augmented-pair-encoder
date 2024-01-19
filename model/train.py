import logging
import os
from datetime import datetime
from math import ceil
from typing import List

from fire import Fire
from model.evaluation import CorrelationEvaluator
from model.model import PairEncoder
from model.util import PairInput
from model.weak_supervision import label_sentences
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


def train_encoder(
    train_samples: List[PairInput],
    evaluator: CorrelationEvaluator = None,
    model_name: str = "cross-encoder/stsb-roberta-base",
    similarity_model: str = "intfloat/e5-base-v2",
    batch_size: int = 32,
    learning_rate=8e-5,
    epochs=10,
    eval_steps=0,  # if 0, evaluate after each epoch
    k=0,
    save_to=None,
    verbose=False,
    device="cuda",
):
    encoder = PairEncoder(model_name, device=device)
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging_param = {
        "PairEncoder": model_name,
        "batch": batch_size,
        "lr": learning_rate,
        "epochs": epochs,
        "k": k,
    }
    logging.info(logging_param)

    if k > 0:
        logging.info("Training encoder for 1 epoch prior to weak supervision...")
        dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
        encoder.fit(
            dataloader=dataloader,
            evaluator=evaluator,
            epochs=1,
            warmup_steps=ceil(len(dataloader) * 1 * 0.1),
            learning_rate=learning_rate,
            verbose=verbose,
        )
        logging.info("Weakly labeling sentences...")
        sent_transformer = SentenceTransformer(similarity_model)
        weak_samples = label_sentences(
            sent_transformer=sent_transformer,
            encoder=encoder,
            train_samples=train_samples,
            top_k=k,
            batch_size=batch_size,
            device=device,
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