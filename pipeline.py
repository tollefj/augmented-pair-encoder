from datasets import load_dataset
from fire import Fire
from model.data import process_sts
from model.evaluation import CorrelationEvaluator
from model.train import train_encoder
from scipy.stats import spearmanr


def run(
    model_name: str,
    similarity_model: str,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    eval_steps: int = 0,
    k: int = 0,
    verbose: bool = False,
    save_to: str = "output",
    device: str = "cuda",
    dataset_name: str = "mteb/stsbenchmark-sts",
):
    dataset = load_dataset(dataset_name)
    train = process_sts(dataset["train"])
    test = process_sts(dataset["test"])
    dev = process_sts(dataset["validation"])

    evaluator = CorrelationEvaluator.load(dev)

    encoder = train_encoder(
        train_samples=train,
        evaluator=evaluator,
        model_name=model_name,
        similarity_model=similarity_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        eval_steps=eval_steps,
        k=k,
        verbose=verbose,
        save_to=save_to,
        device=device,
    )

    test_data = [(data.pair[0], data.pair[1]) for data in test]
    test_scores = [p.label for p in test]

    preds = encoder.predict(test_data)
    corr = spearmanr(test_scores, preds).correlation
    print(f"Spearman correlation on test set: {corr}")


if __name__ == "__main__":
    Fire(run)
