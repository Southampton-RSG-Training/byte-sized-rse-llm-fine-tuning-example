# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "click",
#   "datasets",
#   "tqdm",
#   "transformers",
# ]
# ///

"""Download Models and Data

This script downloads models and data to the local transformers cache.

This needs to be run before doing any fine-tuning of models.
"""

from getpass import getuser
from pathlib import Path
import os

import click
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "yelp/yelp_review_full"
DEFAULT_CACHE = Path("/scratch") / getuser() / "hf_cache"


def get_llm(repo_name, trust_remote=False, auto_model=AutoModelForCausalLM):
    """Download a model from a huggingface repository."""
    tokenizer = AutoTokenizer.from_pretrained(
        repo_name,
        trust_remote_code=trust_remote,
    )
    model = auto_model.from_pretrained(
        repo_name,
        trust_remote_code=trust_remote,
    )
    return tokenizer, model


@click.command()
@click.option("--model", default=DEFAULT_MODEL, help="path to the model")
@click.option("--dataset", default=DEFAULT_DATASET, help="path to the dataset")
@click.option("--trust", is_flag=True, default=False, help="trust remote code")
@click.option("--cache-dir", default=DEFAULT_CACHE, help="cache directory to use for weights")
def main(model, dataset, trust, cache_dir):
    if cache_dir.exists() and cache_dir.is_dir():
        os.environ["HF_HOME"] = os.fspath(cache_dir)
    get_llm(model, trust)
    print(load_dataset(dataset).num_rows)


if __name__ == "__main__":
    main()
