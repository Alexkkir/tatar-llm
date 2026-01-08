# Tatar GPT-2 (Small)

This project trains a small GPT-2 model from scratch on the Tatar language using the Wikipedia dataset.

## Project Structure

*   `download_data.py`: Downloads the Tatar Wikipedia dataset using Hugging Face Datasets.
*   `train_tokenizer.py`: Trains a Byte-Level BPE tokenizer on the collected data.
*   `train.py`: Trains a GPT-2 model (decoder-only) from scratch.
*   `inference.py`: Demonstrates the model's capabilities by generating text from prompts.
*   `data/`: Contains raw text and tokenizer files.
*   `tatar_gpt_model/`: Contains the saved trained model and tokenizer.

## Steps to Reproduce

1.  **Install Dependencies**:
    The project uses `uv`.
    ```bash
    uv sync
    ```

2.  **Download Data**:
    ```bash
    uv run python download_data.py
    ```
    This downloads ~100MB of Tatar text from Wikipedia.

3.  **Train Tokenizer**:
    ```bash
    uv run python train_tokenizer.py
    ```
    Creates a BPE tokenizer with a vocabulary size of 32,000.

4.  **Train Model**:
    ```bash
    uv run python train.py
    ```
    Trains a small GPT-2 model for 3 epochs.
    *   Context length: 512
    *   Parameters: ~20M
    *   Mixed Precision: fp16 enabled

5.  **Run Inference**:
    ```bash
    uv run python inference.py
    ```
    Generates text samples.

## Results

The model achieves a training loss of ~2.96 after 3 epochs.
Example generations:

*   **Prompt**: "Татарстан - ул"
    **Output**: "Татарстан - ул Татарстан Республикасының Кайбыч районындагы авыл..." (Grammatically correct Tatar sentence structure).

*   **Prompt**: "Казан шәһәре"
    **Output**: "Казан шәһәре () — Россия Федерациясенең субъекты..."

## Model Details

*   **Architecture**: GPT-2 (Decoder-only transformer)
*   **Vocab Size**: 32,000
*   **Context Window**: 512
*   **Optimization**: AdamW, Linear Warmup, fp16 mixed precision.

