"""
Multilingual Tokenizer Analysis Tool

A comprehensive tool for analyzing and comparing different tokenizers' performance
on multilingual text data, with a focus on English-Hindi language pairs.

This script evaluates tokenization efficiency, vocabulary coverage, and cost implications
across multiple tokenizers, providing visualizations and detailed metrics.
"""

import time
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import MBart50TokenizerFast, AutoTokenizer
from collections import defaultdict
import colorama
from colorama import Fore, Back, Style

from plotting import *

# Initialize colorama to work on all platforms
colorama.init(autoreset=True)


def setup_directories():
    """
    Create directories for output files.

    Returns
    -------
    tuple
        A tuple containing the paths to the results and plots directories.
    """
    dirs = ["results", "plots"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    return dirs[0], dirs[1]


def load_data(dataset_name="higashi1/challenge_enHindi", split="train", sample_size=500):
    """
    Load a parallel dataset and sample a subset for analysis.

    Parameters
    ----------
    dataset_name : str, optional
        The name of the Hugging Face dataset, by default "higashi1/challenge_enHindi"
    split : str, optional
        The dataset split to use, by default "train"
    sample_size : int, optional
        Number of examples to sample for detailed analysis, by default 500

    Returns
    -------
    tuple
        Tuple of source samples and target samples
    """
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split=split)
    src_texts = ds["en"]
    tgt_texts = ds["hi"]

    # Sample a subset for detailed analysis
    sample_size = min(sample_size, len(src_texts))
    sample_indices = np.random.choice(len(src_texts), sample_size, replace=False)
    src_sample = [src_texts[i] for i in sample_indices]
    tgt_sample = [tgt_texts[i] for i in sample_indices]

    print(f"Dataset loaded. Using {sample_size} samples for analysis.")
    return src_sample, tgt_sample


def initialize_tokenizers():
    """
    Initialize multiple tokenizers for comparison.

    Returns
    -------
    dict
        Dictionary of tokenizer name -> tokenizer object
    """
    print("Initializing tokenizers...")
    return {
        "mBART-50": MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt"),
        "IndicBERT": AutoTokenizer.from_pretrained('ai4bharat/indic-bert'),
        "SUTRA": AutoTokenizer.from_pretrained("TWO/sutra-mlt256-v2"),
    }


def analyze_vocabulary_sizes(tokenizers):
    """
    Analyze and compare vocabulary sizes across tokenizers.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object

    Returns
    -------
    dict
        Dictionary with vocabulary size information
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Vocabulary sizes:{Style.RESET_ALL}")
    vocab_sizes = {}

    for name, tok in tokenizers.items():
        vocab_size = getattr(tok, "vocab_size", len(tok.get_vocab()) if hasattr(tok, "get_vocab") else "N/A")
        vocab_sizes[name] = vocab_size
        print(f"   • {name:15s}: {vocab_size:,} tokens")

    return vocab_sizes


def token_per_word_ratio(tokenizer, sentences):
    """
    Calculate the ratio of tokens to words for a set of sentences.

    Parameters
    ----------
    tokenizer : object
        The tokenizer to use
    sentences : list
        List of sentences to analyze

    Returns
    -------
    float
        Float representing the ratio of tokens to words
    """
    total_subwords = sum(len(tokenizer.tokenize(sent)) for sent in sentences)
    total_words = sum(len(sent.split()) for sent in sentences)
    return total_subwords / total_words


def analyze_token_word_ratios(tokenizers, src_sample, tgt_sample):
    """
    Analyze token-per-word ratios for source and target languages.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object
    src_sample : list
        List of source language sentences
    tgt_sample : list
        List of target language sentences

    Returns
    -------
    list
        List of tuples with tokenizer name, language, and ratio
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Token-per-word ratio (higher = less efficient):{Style.RESET_ALL}")
    ratio_data = []

    for name, tok in tokenizers.items():
        en_ratio = token_per_word_ratio(tok, src_sample)
        hi_ratio = token_per_word_ratio(tok, tgt_sample)
        ratio_data.append((name, "English", en_ratio))
        ratio_data.append((name, "Hindi", hi_ratio))
        print(f"   • {name:15s}: English: {en_ratio:.2f} | Hindi: {hi_ratio:.2f} | Hindi/English: {hi_ratio/en_ratio:.2f}x")

    return ratio_data


def get_token_length_stats(tokenizer, sentences):
    """
    Calculate statistics about token lengths for sentences.

    Parameters
    ----------
    tokenizer : object
        The tokenizer to use
    sentences : list
        List of sentences to analyze

    Returns
    -------
    dict
        Dictionary with statistics about token lengths
    """
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    return {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "p90": np.percentile(lengths, 90),
        "max": np.max(lengths),
        "all_lengths": lengths
    }


def analyze_token_length_distribution(tokenizers, src_sample, tgt_sample):
    """
    Analyze token length distribution for source and target languages.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object
    src_sample : list
        List of source language sentences
    tgt_sample : list
        List of target language sentences

    Returns
    -------
    dict
        Dictionary with token length statistics
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Token length distribution (input length impacts):{Style.RESET_ALL}")
    token_length_stats = {}

    for name, tok in tokenizers.items():
        en_stats = get_token_length_stats(tok, src_sample)
        hi_stats = get_token_length_stats(tok, tgt_sample)
        token_length_stats[name] = {"en": en_stats, "hi": hi_stats}
        print(f"   • {name:15s}:")
        print(f"     - English: mean={en_stats['mean']:.1f}, median={en_stats['median']:.1f}, p90={en_stats['p90']:.1f}")
        print(f"     - Hindi:   mean={hi_stats['mean']:.1f}, median={hi_stats['median']:.1f}, p90={hi_stats['p90']:.1f}")
        print(f"     - Expansion ratio (Hindi/English): {hi_stats['mean']/en_stats['mean']:.2f}x")

    return token_length_stats


def chars_per_token(tokenizer, sentences):
    """
    Calculate the average number of characters per token.

    Parameters
    ----------
    tokenizer : object
        The tokenizer to use
    sentences : list
        List of sentences to analyze

    Returns
    -------
    float
        Float representing the average number of characters per token
    """
    total_chars = sum(len(sent) for sent in sentences)
    total_tokens = sum(len(tokenizer.encode(sent, add_special_tokens=False)) for sent in sentences)
    return total_chars / total_tokens if total_tokens > 0 else 0


def analyze_character_efficiency(tokenizers, src_sample, tgt_sample):
    """
    Analyze character efficiency for source and target languages.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object
    src_sample : list
        List of source language sentences
    tgt_sample : list
        List of target language sentences

    Returns
    -------
    dict
        Dictionary with character efficiency information
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Character efficiency (chars per token - higher is better):{Style.RESET_ALL}")
    character_efficiency = {}

    for name, tok in tokenizers.items():
        en_cpt = chars_per_token(tok, src_sample)
        hi_cpt = chars_per_token(tok, tgt_sample)
        character_efficiency[name] = {"en": en_cpt, "hi": hi_cpt}
        print(f"   • {name:15s}: English: {en_cpt:.2f} | Hindi: {hi_cpt:.2f}")

    return character_efficiency


def analyze_tokenization_example(tokenizers, example_text):
    """
    Analyze tokenization of a specific example text.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object
    example_text : str
        Text to analyze

    Returns
    -------
    dict
        Dictionary with tokenization examples
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Sample tokenization comparison:{Style.RESET_ALL}")
    print(f"\nOriginal text ({len(example_text)} chars, {len(example_text.split())} words):")
    print(example_text)

    token_examples = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(example_text)
        token_examples[name] = tokens
        print(f"\n{name} ({len(tokens)} tokens, {len(tokens)/len(example_text.split()):.2f} tokens/word):")
        print(" ".join(tokens[:20]) + "..." if len(tokens) > 20 else " ".join(tokens))

    return token_examples

def summarize_tokenization_efficiency(token_length_stats):
    """
    Summarize tokenization efficiency findings.

    Parameters
    ----------
    token_length_stats : dict
        Dictionary with token length statistics

    Returns
    -------
    dict
        Dictionary with summary information
    """
    print("\n→ SUMMARY OF TOKENIZATION EFFICIENCY:")

    best_for_english = min([(name, token_length_stats[name]["en"]["mean"])
                          for name in token_length_stats.keys()], key=lambda x: x[1])

    best_for_hindi = min([(name, token_length_stats[name]["hi"]["mean"])
                        for name in token_length_stats.keys()], key=lambda x: x[1])

    most_balanced = min([(name, abs(token_length_stats[name]["hi"]["mean"]/token_length_stats[name]["en"]["mean"] - 1))
                        for name in token_length_stats.keys()], key=lambda x: x[1])

    print(f"• Most efficient for English: {best_for_english[0]} ({best_for_english[1]:.1f} tokens/sentence)")
    print(f"• Most efficient for Hindi: {best_for_hindi[0]} ({best_for_hindi[1]:.1f} tokens/sentence)")
    print(f"• Most balanced across languages: {most_balanced[0]} (diff from 1:1 ratio: {most_balanced[1]:.2f})")
    print("\nRecommendation: Consider using the most efficient tokenizer for your primary target language,")
    print("or the most balanced one for a multilingual application.")

    return {
        "best_for_english": best_for_english,
        "best_for_hindi": best_for_hindi,
        "most_balanced": most_balanced
    }


def analyze_token_efficiency(tokenizers, tgt_sample, sample_size=100):
    """
    Analyze token efficiency for specific words.

    Parameters
    ----------
    tokenizers : dict
        Dictionary of tokenizer name -> tokenizer object
    tgt_sample : list
        List of target language sentences
    sample_size : int, optional
        Number of sentences to analyze, by default 100

    Returns
    -------
    dict
        Dictionary with inefficient words
    """
    print(f"\n{Fore.BLUE}{Style.BRIGHT}→ Token efficiency analysis:{Style.RESET_ALL}")
    # Let's analyze a sample of sentences to identify words that get split into many tokens
    inefficient_words = defaultdict(list)

    # Take a small sample for detailed analysis
    analysis_sample = tgt_sample[:sample_size]  # Just use a limited number of sentences

    for sent in analysis_sample:
        words = sent.split()
        for word in words:
            for name, tokenizer in tokenizers.items():
                tokens = tokenizer.tokenize(word)
                if len(tokens) > 1:  # If the word is split into multiple tokens
                    inefficient_words[name].append((word, len(tokens), tokens))

    # Find the most inefficient words for each tokenizer
    result = {}
    for name, word_data in inefficient_words.items():
        if not word_data:
            continue
        # Sort by number of tokens (most inefficient first)
        word_data.sort(key=lambda x: x[1], reverse=True)
        result[name] = word_data[:5]  # Top 5 least efficient words
        print(f"\n{Fore.YELLOW}{name}{Style.RESET_ALL} - Top 5 least efficiently tokenized Hindi words:")
        for word, num_tokens, tokens in result[name]:
            print(f"   • '{word}' → {num_tokens} tokens: {' '.join(tokens)}")

    return result


def save_results_to_csv(results, results_dir):
    """
    Save analysis results to CSV files.

    Parameters
    ----------
    results : dict
        Dictionary with analysis results
    results_dir : str
        Directory to save results

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    saved_files = {}

    # Save token-per-word ratios
    ratio_df = pd.DataFrame(results["ratio_data"], columns=["Tokenizer", "Language", "Tokens per Word"])
    ratio_path = os.path.join(results_dir, "token_per_word_ratios.csv")
    ratio_df.to_csv(ratio_path, index=False)
    saved_files["token_per_word_ratios"] = ratio_path

    # Save token length statistics
    token_length_rows = []
    for tokenizer, stats in results["token_length_stats"].items():
        for lang, metrics in stats.items():
            token_length_rows.append({
                "Tokenizer": tokenizer,
                "Language": "English" if lang == "en" else "Hindi",
                "Mean": metrics["mean"],
                "Median": metrics["median"],
                "90th Percentile": metrics["p90"],
                "Max": metrics["max"]
            })
    token_length_df = pd.DataFrame(token_length_rows)
    token_length_path = os.path.join(results_dir, "token_length_stats.csv")
    token_length_df.to_csv(token_length_path, index=False)
    saved_files["token_length_stats"] = token_length_path

    print(f"\n{Fore.GREEN}Results saved to CSV files in {results_dir}/{Style.RESET_ALL}")
    for name, path in saved_files.items():
        print(f"  • {name}: {os.path.basename(path)}")

    return saved_files


def main():
    """
    Main function to run the tokenization analysis.

    This function coordinates the entire analysis process:
    1. Setting up directories
    2. Loading data
    3. Initializing tokenizers
    4. Running analyses
    5. Generating plots
    6. Summarizing findings
    7. Saving results
    """
    start_time = time.time()

    print(f"{Back.WHITE}{Fore.BLACK}{Style.BRIGHT} MULTILINGUAL TOKENIZER ANALYSIS TOOL {Style.RESET_ALL}")

    # Create directories for output
    results_dir, plots_dir = setup_directories()

    # Load and sample data
    src_sample, tgt_sample = load_data(sample_size=500)

    # Initialize tokenizers
    tokenizers = initialize_tokenizers()

    # Run analyses
    vocab_sizes = analyze_vocabulary_sizes(tokenizers)
    ratio_data = analyze_token_word_ratios(tokenizers, src_sample, tgt_sample)
    token_length_stats = analyze_token_length_distribution(tokenizers, src_sample, tgt_sample)
    character_efficiency = analyze_character_efficiency(tokenizers, src_sample, tgt_sample)

    # Analyze a specific example text
    example_hi = "जी हां। आम बोलचाल में, विभिन्न अन्य अधिकार क्षेत्र के प्रतिनिधि / ट्रस्ट रहित कानूनों के अंतर्गत उपलब्ध सीबीआई की न्यूनतर अर्थदंड व्यवस्था के इस तरह के कार्यक्रमों / व्यवस्थाओं का उल्लेख किया गया है और उदारता कार्यक्रम' के रूप में जाना जाता है"
    token_examples = analyze_tokenization_example(tokenizers, example_hi)

    # Summarize findings
    summary = summarize_tokenization_efficiency(token_length_stats)

    # Analyze token efficiency for specific words
    inefficient_words = analyze_token_efficiency(tokenizers, tgt_sample)

    # Save results to CSV
    results = {
        "vocab_sizes": vocab_sizes,
        "ratio_data": ratio_data,
        "token_length_stats": token_length_stats,
        "character_efficiency": character_efficiency,
        "token_examples": token_examples,
        "summary": summary,
        "inefficient_words": inefficient_words
    }
    saved_files = save_results_to_csv(results, results_dir)
    plot_token_length_distribution_plotly(token_length_stats, plots_dir)
    plot_token_per_word_plotly(ratio_data, plots_dir)

    # Print execution time
    execution_time = time.time() - start_time
    print(f"\n{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} Analysis completed in {execution_time:.2f} seconds {Style.RESET_ALL}")


if __name__ == "__main__":
    main()