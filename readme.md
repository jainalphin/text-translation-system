# Multilingual Translator and Tokenizer Analysis
This repository contains tools for multilingual translation (English-Hindi) and tokenizer analysis. The project includes two main components:

1. **Multilingual Translator**: Fine-tunes and evaluates neural machine translation models
2. **Tokenizer Analysis Tool**: Compares different tokenizers' performance on multilingual text

## Setup and Installation
```commandline
cd text-translation-system
pip install -r requirements.txt
```

## Dataset Download
Use a sample English-Hindi dataset for experiments.

Download it using:

```commandline
pip install gdown
gdown https://drive.google.com/uc?id=1fEnWm-S0-5dpHY3nRxegAA0ofAIDY180 -O "all_data_en_hin.csv"
```

## Usage

### 1. Token Analysis
Navigate to token_analysis/ and run:

```
python token_analysis.py
```

This will:

* Download a small Hindi-English dataset from HuggingFace (higashi1/challenge_enHindi).
* Initialize multiple tokenizers. 
* Analyze vocabulary sizes, token-to-word ratios, token length distributions. 
* Save detailed CSV reports and generate plots.

### 2. Translation (Training & Inference)
Navigate to training/ and run ```translator.py```
*  For detailed instructions on translation tasks, please refer to the [readme.md](training/readme.md)

## Report:
For a detailed analysis report, you can view it here:
[Detailed Report](https://drive.google.com/file/d/186oOOCRAb1og_IF3IQwqrObA_GX163gI/view?usp=sharing)