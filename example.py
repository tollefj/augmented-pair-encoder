from pipeline import run

run(
    model_name="cross-encoder/stsb-roberta-base",
    similarity_model="intfloat/multilingual-e5-base",
    learning_rate=8e-5,
    epochs=1,
    eval_steps=100,
    dataset_name="mteb/stsbenchmark-sts",
    k=0,
    verbose=True,
    save_to=None,
)
