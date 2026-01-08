# Tatar GPT-2 (Small)

This project trains a small GPT-2 model from scratch on the Tatar language using the Wikipedia dataset.

### Artifacts
- [Data](https://huggingface.co/datasets/alexkkir/tatar-wikipedia-corpus)
- [Trained tokenizer & Model](https://huggingface.co/alexkkir/tatar-gpt2-small/tree/main)

## Project Structure

*   `download_data.py`: Downloads the Tatar Wikipedia dataset using Hugging Face Datasets.
*   `train_tokenizer.py`: Trains a Byte-Level BPE tokenizer on the collected data.
*   `train.py`: Trains a GPT-2 model (decoder-only) from scratch.
*   `inference.py`: Demonstrates the model's capabilities by generating text from prompts.
*   `data/`: Contains raw text and tokenizer files.
*   `tatar_gpt_model/`: Contains the saved trained model and tokenizer.



## Steps to Reproduce

```
uv sync
uv run python download_data.py
uv run python train_tokenizer.py
uv run python train.py
uv run python inference.py
```
