# English-Hindi Legal Document Translator and Tokenizer Analysis
This repository contains tools for legal document translation (English-Hindi) and tokenizer analysis. The project includes two main components:

1. **Machine Translator**: Fine-tunes and evaluates neural machine translation models
2. **Tokenizer Analysis**: Compares different tokenizers' performance on multilingual text


## Setup and Installation
```commandline
git clone https://github.com/yourusername/text-translation-system.git

# Change to project directory
cd text-translation-system

# Install dependencies
pip install -r requirements.txt
```

## Dataset Download
Use a sample English-Hindi dataset for experiments.

```commandline
# Install gdown for Google Drive downloads
pip install gdown

# Download the dataset
gdown https://drive.google.com/uc?id=1fEnWm-S0-5dpHY3nRxegAA0ofAIDY180 -O "all_data_en_hin.csv"
```

## Usage

### 1. Token Analysis
Navigate to tokenizer_analysis/ directory and run:

```
python tokenizer_comparison.py
```

This will:

* Download a small Hindi-English dataset from HuggingFace (higashi1/challenge_enHindi).
* Initialize multiple tokenizers. 
* Analyze vocabulary sizes, token-to-word ratios, token length distributions. 
* Save detailed CSV reports and generate plots.

### 2. Translation (Training & Inference)
Navigate to translation/ directory and run:
```commandline
python model_trainer.py --mode pipeline --data_file all_data_en_hin.csv --source_lang en --target_langs hi --output_dir ./my_translator_model
```
For detailed instructions on all translation tasks and options, please refer to the [Training Guide](translation/training_guide.md)

## 3. Jupyter Notebooks
* [English-Hindi Training Notebook](notebooks/en_hi_model_training.ipynb)
* [English-Hindi Inference Notebook](notebooks/en_hi_model_inference.ipynb)


## Report:
For a detailed analysis report, you can view it here:
[Detailed Analysis Report](https://drive.google.com/file/d/1H2AmB4-F2gnGy_pDVL44EL7VHROfwU5X/view?usp=sharing)
