# byte-sized-rse-llm-fine-tuning-example

Example code for example of running LLM fine-tuning workflows using SLURM, Accelerate and Transformers.

This fine-tunes an LLM to perform a classification task using the Yelp dataset. It should work as a model for other, similar fine-tuning tasks.

## Installation

On the login node, create a virtual environment and use pip to install the dependencies.

```bash
load module python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run `llm_download.py` to get the dataset and model weights.

```bash
python llm_download.py
```

For the large FSDP example, you will need to download Qwen3 weights as well:

```bash
python llm_download.py --model Qwen/Qwen3-8B
```

## Running

You can verify that the code works correctly by running a couple of epochs on the login node with:

```bash
python llm_fine_tuning.py
```

Assuming this works, you can run the code using sbatch.  There are 4 example scripts:

- `single_node_l4.sh` - basic data parallel fine-tuning on a single `l4` node with two GPUs
- `single_node_fsdp.sh` - fully-sharded data parallel fine-tuning on a single `l40` node with two GPUs
- `single_node_large_fsdp.sh` - fully-sharded data parallel fine-tuning on a single `a100` node with two GPUs (requires downloading Qwen3-8B)
- `multi_node_iridis7.sh` - data parallel multi-node training on two `i7_h200` nodes.
