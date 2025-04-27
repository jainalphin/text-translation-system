# Multilingual Translator

## Installation

### Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/multilingual-translator.git
   cd multilingual-translator
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Format

Your data file should be a CSV with columns for each language. For example:

```
en,hi
Hello world,नमस्ते दुनिया
How are you?,आप कैसे हैं?
```

The column names should match the language codes you provide in the command line arguments.

## Usage

### Data Splitting

To split a single data file into train, validation, and test sets:

```bash
python translator.py --mode split-data --data_file data.csv --output_dir ./data_splits --val_size 0.15 --test_size 0.15
```

### Fine-tuning

Fine-tune the model with specific train and validation sets:

```bash
python translator.py --mode finetune \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --train_file ./data_splits/train.csv \
  --val_file ./data_splits/val.csv \
  --source_lang en \
  --target_langs hi \
  --output_dir ./finetuned_model \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-5
```

Or provide a single data file to be automatically split:

```bash
python translator.py --mode finetune \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --data_file data.csv \
  --source_lang en \
  --target_langs hi \
  --output_dir ./finetuned_model \
  --epochs 3
```

### Single Text Translation

Translate a single text:

```bash
python translator.py --mode translate \
  --model ./finetuned_model/final \
  --source_lang en \
  --target_langs hi \
  --text "Hello, how are you doing today?"
```

### Evaluation

Evaluate translation quality using reference translations:

```bash
python translator.py --mode evaluate \
  --model ./finetuned_model/final \
  --test_file ./data_splits/test.csv \
  --source_lang en \
  --target_langs hi \
  --output_dir ./evaluation_results
```

### Complete Pipeline

Run the complete pipeline from fine-tuning to evaluation:

```bash
python translator.py --mode pipeline \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --data_file data.csv \
  --source_lang en \
  --target_langs hi \
  --output_dir ./complete_pipeline \
  --epochs 3 \
  --batch_size 4
```

Or with pre-split data:

```bash
python translator.py --mode pipeline \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --train_file ./data_splits/train.csv \
  --val_file ./data_splits/val.csv \
  --test_file ./data_splits/test.csv \
  --source_lang en \
  --target_langs hi \
  --output_dir ./complete_pipeline \
  --epochs 3
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | Operation mode: `finetune`, `translate`, `batch-translate`, `evaluate`, `pipeline`, or `split-data` |
| `--model` | Pretrained model name or path to fine-tuned model |
| `--source_lang` | Source language code (e.g., `en`) |
| `--target_langs` | Comma-separated target language codes (e.g., `hi`) |
| `--data_file` | Path to single data CSV file for splitting |
| `--train_file` | Path to training data CSV (if already split) |
| `--val_file` | Path to validation data CSV (if already split) |
| `--test_file` | Path to test data CSV (if already split) |
| `--val_size` | Validation set size as a fraction of total data (default: 0.15) |
| `--test_size` | Test set size as a fraction of total data (default: 0.15) |
| `--random_seed` | Random seed for data splitting (default: 42) |
| `--output_dir` | Directory to save the fine-tuned model/results |
| `--output_file` | File to save translations |
| `--max_samples` | Maximum number of examples to use for training |
| `--epochs` | Number of training epochs (default: 1) |
| `--batch_size` | Training batch size (default: 2) |
| `--learning_rate` | Learning rate for training (default: 1e-5) |
| `--text` | Text to translate (for translate mode) |

## Supported Languages

The tool supports translation between many languages including:

- English (en)
- Hindi (hi)

## Examples

### Example 1: Quick Translation

Fine-tune a model for English-to-Hindi translation and translate a sentence:

```bash
# First, split your data
python translator.py --mode split-data --data_file en_hi_data.csv --output_dir ./data

# Fine-tune the model
python translator.py --mode finetune --train_file ./data/train.csv --val_file ./data/val.csv --source_lang en --target_langs hi --output_dir ./en_hi_model --epochs 2

# Translate a sentence
python translator.py --mode translate --model ./en_hi_model/final --source_lang en --target_langs hi --text "Machine learning is transforming the world."
```

### Example 2: Complete Pipeline

Run the complete pipeline for multi-target translation (English to Hindi):

```bash
python translator.py --mode pipeline --data_file multi_lang_data.csv --source_lang en --target_langs hi --output_dir ./multilingual_model --epochs 3 --batch_size 4
```

## Tips for Best Results

1. **Data Quality**: Make sure your training data has high-quality translations
2. **GPU Usage**: Using a GPU significantly speeds up training
3. **Batch Size**: Adjust batch size based on your GPU memory
4. **Learning Rate**: Start with a small learning rate (1e-5 to 5e-5)
5. **Pre-processing**: The script handles basic pre-processing, but clean data works best
6. **Multiple Epochs**: For better results, train for multiple epochs (3-5)

## Troubleshooting

- **Out of Memory**: Reduce batch size or use gradient accumulation
- **Slow Training**: Make sure you're using a GPU; reduce max sequence length
- **Poor Translation Quality**: Use more training data, increase epochs, check data quality


## Acknowledgements

- This project builds on the [mBART model](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) from Facebook/Meta AI Research