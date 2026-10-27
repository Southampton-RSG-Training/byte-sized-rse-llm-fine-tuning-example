# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "accelerate",
#   "click",
#   "datasets",
#   "torch",
#   "transformers",
# ]
# ///

"""Download Models and Data

This script downloads models and data to the local transformers cache.

This needs to be run before doing any fine-tuning of models.
"""


from accelerate import Accelerator
import click
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "yelp/yelp_review_full"


def prepare_dataset(
    dataset_path,
    tokenizer,
    subset=1,
    splits={"test": "test", "train": "train"},
    columns={"text": "text", "labels": "label"}
):
    # XXX this is slightly over-engineered

    print("Loading the data")

    # load the dataset with the option of only taking a percentage subset of it
    dataset = load_dataset(
        dataset_path,
        split={
            'train': f'{splits["train"]}[:{int(subset*100)}%]',
            'test': f'{splits["test"]}[:{int(subset*100)}%]',
        },
    )

    print("Normalizing the data")

    # normalize teat/train splits
    if not isinstance(dataset, DatasetDict):
        dataset = dataset.test_train_split(shuffle=True)
    # elif set(dataset.keys()) != {'test', 'train'}:
    #     dataset = DatasetDict({
    #         'test': dataset[splits['test']],
    #         'train': dataset[splits['train']],
    #     })

    # take subset of the data
    # if 0 < subset < 1:
    #     print(f"Taking {subset:0.1%} of the data")
    #     for split, num_rows in dataset.num_rows.items():
    #         # turn proportion to approximate index
    #         n = int(num_rows * subset)
    #         dataset[split] = Dataset(dataset[split][:n])

    # normalize columns

    print("Tokenizing the text")

    def tokenize_function(examples):
        return tokenizer(
            examples[columns["text"]],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns([columns["text"]])
    tokenized_datasets = tokenized_datasets.rename_column(columns["labels"], "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def prepare(model_path, dataset_path, subset=1, splits={"test": "test",
"train": "train"}):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = prepare_dataset(dataset_path, tokenizer, subset)
    num_labels = len(dataset.unique('labels')['train'])
    print(f"num_labels {num_labels}")

    print("Instantiating the Dataloader")

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(dataset["test"], batch_size=8)

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )

    return model, train_dataloader, eval_dataloader


def train(accelerator, model, train, eval_dataloader, num_epochs=10):
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    num_epochs = 10
    num_training_steps = num_epochs * len(train)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # The model, optimizer and data loaders are passed to the Accelerator instance
    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = \
        accelerator.prepare(
            train, eval_dataloader, model, optimizer, lr_scheduler
        )

    print("Begin training...")

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        accelerator.print(f"Epoch: {epoch}")

    print("Training done...")

@click.command()
@click.option("--model", default=DEFAULT_MODEL, help="path to the model")
@click.option("--dataset", default=DEFAULT_DATASET, help="path to the dataset")
@click.option("--output-dir", default="fine_tuned_classifier", help="path to the
 output location")
@click.option("--num-epochs", default=10, help="the number of epochs")
@click.option("--subset", default=0.01, help="the proportion of the dataset to use")
def main(model, dataset, subset, output_dir, num_epochs):
    print("Started")

    accelerator = Accelerator()
    print(f"Device: {accelerator.device}")

    model, train_loader, eval_loader = prepare(model, dataset, subset=subset)
    print("Data loaded.")
    train(accelerator, model, train_loader, eval_loader, num_epochs)

    accelerator.save_state(output_dir)
    print(f"Model saved to: {output_dir}")
    accelerator.end_training()

if __name__ == '__main__':
    main()
