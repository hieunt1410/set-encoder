from lightning_ir import CrossEncoderModule, LightningIRDataModule, LightningIRTrainer, ReRankCallback, RunDataset

# Define the model
module = CrossEncoderModule(
    model_name_or_path="webis/bert-bi-encoder",
    evaluation_metrics=["f1@1"],
)

# Define the data module
data_module = LightningIRDataModule(
    inference_datasets=[
        RunDataset("./runs/msmarco-passage-trec-dl-2019-judged.run"),
        RunDataset("./runs/msmarco-passage-trec-dl-2020-judged.run"),
    ],
    inference_batch_size=4,
)

# Define the search callback
callback = ReRankCallback(save_dir="./re-ranked-runs")

# Define the trainer
trainer = LightningIRTrainer(callbacks=[callback])

# Retrieve relevant documents
trainer.re_rank(module, data_module)