# Bhojpuri POS Tagging with Transformers

A clean, structured project for training and evaluating Part-of-Speech (POS) tagging models specifically for Bhojpuri.

## Project Structure
- `data/`: Datasets and preprocessed files.
- `models/`: Saved model checkpoints.
- `notebooks/`: Exploratory Data Analysis and Colab integration.
- `scripts/`: Production-ready training and preprocessing scripts.
- `utils/`: Reusable helper functions and metrics.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
To train the model, run:
```bash
python scripts/train.py
```
