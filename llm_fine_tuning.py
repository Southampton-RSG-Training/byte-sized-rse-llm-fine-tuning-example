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
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "yelp/yelp_review_full"


def prepare_dataset(
    dataset_path,
    tokenizer,
    subset=1,
    splits={"test": "test", "train": "train"},
):
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

    print("Tokenizing the text")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            pad="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def get_trainer(model_path, dataset_path, training_args, subset):
    """Prepare the model, dataset and build the trainer."""

    print("Loading the tokenizer weights")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = prepare_dataset(dataset_path, tokenizer, subset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading the model weights")

    num_labels = len(dataset.unique('label')['train'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True,
        dtype=torch.float32,  # issue with bfloat16 on Iridis
    )

    print("Preparing the trainer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    return trainer


@click.command()
@click.option("--model", default=DEFAULT_MODEL, help="path to the model")
@click.option("--dataset", default=DEFAULT_DATASET, help="path to the dataset")
@click.option("--output-dir", default="fine_tuned_classifier", help="path to the output location")
@click.option("--num-epochs", default=2, help="the number of epochs")
@click.option("--subset", default=0.01, help="the proportion of the dataset to use")
def main(model, dataset, subset, output_dir, num_epochs):
    print("Started")

    accelerator = Accelerator()
    print(f"Device: {accelerator.device}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    trainer = get_trainer(model, dataset, training_args, subset=subset)
    trainer.train()


if __name__ == '__main__':
    main()
