import os
import argparse
import pandas as pd
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import re
import evaluate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import sys
from datetime import datetime


# Configure logging
def setup_logger(log_file=None, logger_level=logging.DEBUG):
    """Set up and configure logger"""
    logger = logging.getLogger("translator")
    logger.setLevel(logger_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MultilingualTranslator:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", logger=None):
        # Set environment variable to disable W&B
        os.environ['WANDB_DISABLED'] = "True"
        self.model_name = model_name

        # Setup logger
        self.logger = logger

        # Load the model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Load model from pretrained or local path
        try:
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.logger.info(f"Successfully loaded model from {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

        # Language codes for mBART
        self.lang_codes = {
            'hi': 'hi_IN',
            'en': 'en_XX',
        }

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = " ".join(text.split())
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\[\]', '', text)
        return text.strip()

    def preprocess_dataset(self, df, source_lang, target_langs):
        """Preprocess dataset for training"""
        self.logger.info(f"Preprocessing dataset with {len(df)} examples...")

        temp_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            source_text = self.preprocess_text(row[source_lang])
            target_texts = [self.preprocess_text(row[lang]) for lang in target_langs]

            if not source_text or any(not text for text in target_texts):
                skipped += 1
                continue

            concatenated_targets = " [SEP] ".join(target_texts)
            self.tokenizer.src_lang = self.lang_codes.get(source_lang, 'en_XX')

            try:
                encoded_source = self.tokenizer(
                    source_text,
                    max_length=300,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )

                encoded_target = self.tokenizer(
                    concatenated_targets,
                    max_length=300,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )

                temp_data['input_ids'].append(encoded_source['input_ids'][0])
                temp_data['attention_mask'].append(encoded_source['attention_mask'][0])
                temp_data['labels'].append(encoded_target['input_ids'][0])
            except Exception as e:
                self.logger.warning(f"Error encoding text: {e}")
                skipped += 1

        self.logger.info(f"Preprocessing complete. Skipped {skipped} examples.")
        return Dataset.from_dict(temp_data)

    def finetune(self, df, val_df, source_lang, target_langs, output_dir="./finetuned_model",
                 epochs=1, batch_size=1, learning_rate=1e-5, max_samples=None,
                 eval_steps=500, save_steps=1000):
        """Fine-tune the model on the provided dataset"""
        # Sample data if needed
        if max_samples and max_samples < len(df):
            self.logger.info(f"Sampling {max_samples} examples from training data")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        if max_samples and max_samples < len(val_df):
            val_df = val_df.sample(n=min(max_samples // 5, len(val_df)), random_state=42).reset_index(drop=True)

        # Preprocess datasets
        train_dataset = self.preprocess_dataset(df, source_lang, target_langs)
        val_dataset = self.preprocess_dataset(val_df, source_lang, target_langs)

        self.logger.info(f"Training with {len(train_dataset)} examples, validating with {len(val_dataset)} examples")

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training setup
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="no",
            eval_steps=eval_steps,
            save_strategy="best",
            save_steps=save_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            no_cuda=(self.device == "cpu"),
            gradient_accumulation_steps=4,
            fp16=(self.device != "cpu"),
            report_to="none",  # Disable W&B
            logging_dir=os.path.join(output_dir, "logs"),
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Fine-tune the model
        self.logger.info("Starting fine-tuning...")
        trainer.train()

        # Save the model and tokenizer
        final_model_path = os.path.join(output_dir, "final")
        os.makedirs(final_model_path, exist_ok=True)

        self.logger.info(f"Saving fine-tuned model to {final_model_path}")
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        # Save training config
        config = {
            "source_language": source_lang,
            "target_languages": target_langs,
            "model_base": self.model_name,
            "training_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

        with open(os.path.join(output_dir, "training_config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        self.logger.info("Fine-tuning complete!")
        return final_model_path

    def translate_text(self, text, source_lang, target_langs):
        """Translate a single text input to multiple target languages"""
        if not text:
            return {lang: "" for lang in target_langs}

        try:
            self.tokenizer.src_lang = self.lang_codes.get(source_lang, 'en_XX')
        except KeyError:
            self.logger.warning(f"Language code {source_lang} not found. Using 'en_XX' as default.")
            self.tokenizer.src_lang = 'en_XX'

        source_text = self.preprocess_text(text)

        encoded_source = self.tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=300,
            padding=True
        )

        # Move model to device if not already there
        self.model.to(self.device)
        encoded_source = {k: v.to(self.device) for k, v in encoded_source.items()}

        # Generate translation
        try:
            generated_ids = self.model.generate(
                input_ids=encoded_source['input_ids'],
                attention_mask=encoded_source['attention_mask'],
                max_length=300,
                num_beams=4,
                early_stopping=True
            )

            concatenated_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            split_outputs = concatenated_output.split(" [SEP] ")

            # Ensure split outputs match target languages
            if len(split_outputs) < len(target_langs):
                split_outputs.extend([""] * (len(target_langs) - len(split_outputs)))
            elif len(split_outputs) > len(target_langs):
                split_outputs = split_outputs[:len(target_langs)]

            return {lang: output.strip() for lang, output in zip(target_langs, split_outputs)}
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return {lang: "" for lang in target_langs}

    def batch_translate(self, df, source_lang, target_langs, output_file=None):
        """Generate translations for a dataset and optionally save to file"""
        self.logger.info(f"Translating {len(df)} examples...")
        results = {lang: [] for lang in target_langs}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Translating"):
            source_text = self.preprocess_text(row[source_lang])

            # Get translations
            translations = self.translate_text(source_text, source_lang, target_langs)

            # Store translations
            for lang, translation in translations.items():
                results[lang].append(translation)

            # Display some examples
            if idx < 3 or idx % (len(df) // 10) == 0:
                self.logger.info(f"----- Example {idx + 1}: -----")
                self.logger.info(f"Source ({source_lang}): {source_text}")
                for lang, translation in translations.items():
                    self.logger.info(f"Translation ({lang}): {translation}")

        # Create output DataFrame
        results_df = pd.DataFrame(results)

        # Add source text to results
        results_df[source_lang] = df[source_lang].apply(self.preprocess_text)

        # Save to file if requested
        if output_file:
            results_df.to_csv(output_file, index=False)
            self.logger.info(f"Translations saved to {output_file}")

        return results_df

    def evaluate_translations(self, df, translations_df, source_lang, target_langs):
        """Evaluate translation quality using BLEU and chrF metrics, skipping errors"""
        self.logger.info("Evaluating translation quality...")
        metrics = {}
        error_examples = {}  # Track examples with errors

        for lang in target_langs:
            if lang not in df.columns:
                self.logger.warning(f"Target language {lang} not in reference data. Skipping evaluation.")
                continue

            # Initialize error tracking for this language
            error_examples[lang] = []
            valid_examples = []  # To track which examples are valid

            all_references = []
            all_predictions = []

            # Process each example and skip problematic ones
            for i, (ref, pred) in enumerate(zip(df[lang], translations_df[lang])):
                try:
                    # Preprocess texts
                    clean_ref = self.preprocess_text(ref) if not pd.isna(ref) else ""
                    clean_pred = pred if not pd.isna(pred) else ""

                    # Skip examples where either reference or prediction is empty
                    if not clean_ref or not clean_pred:
                        error_examples[lang].append({
                            "index": i,
                            "reason": "Empty reference or prediction",
                            "source": df[source_lang].iloc[i],
                            "reference": clean_ref,
                            "prediction": clean_pred
                        })
                        continue

                    # Add to valid examples
                    all_references.append(clean_ref)
                    all_predictions.append(clean_pred)
                    valid_examples.append(i)

                except Exception as e:
                    error_examples[lang].append({
                        "index": i,
                        "reason": str(e),
                        "source": df[source_lang].iloc[i] if i < len(df[source_lang]) else "Unknown"
                    })

            # Skip evaluation if no valid examples
            if not all_references or not all_predictions:
                self.logger.warning(f"No valid examples for language {lang}. Skipping evaluation.")
                metrics[lang] = {
                    "BLEU": 0.0,
                    "chrF": 0.0,
                    "valid_examples": 0,
                    "error_examples": len(error_examples[lang])
                }
                continue

            # Load metrics
            bleu = evaluate.load("bleu")
            chrf = evaluate.load("chrf")

            try:
                # CRITICAL FIX: Format data correctly for the metrics
                # For BLEU: each prediction needs to be tokenized, and each reference needs to be a list of tokenized strings
                preds_for_bleu = [pred.split() for pred in all_predictions]
                refs_for_bleu = [[ref.split()] for ref in
                                 all_references]  # Double list: one list per example, one list per reference

                # For chrF: expects a list of strings for both references and predictions
                refs_for_chrf = all_references
                preds_for_chrf = all_predictions

                # Calculate metrics with correct formatting
                bleu_score = bleu.compute(predictions=preds_for_bleu, references=refs_for_bleu)
                chrf_score = chrf.compute(predictions=preds_for_chrf, references=refs_for_chrf)

                metrics[lang] = {
                    "BLEU": bleu_score["bleu"],
                    "chrF": chrf_score["score"],
                    "valid_examples": len(valid_examples),
                    "error_examples": len(error_examples[lang])
                }

                # Log metrics
                self.logger.info(f"Evaluation for target language: {lang}")
                self.logger.info(f"BLEU score: {bleu_score['bleu']:.4f}")
                self.logger.info(f"chrF score: {chrf_score['score']:.4f}")
                self.logger.info(f"Valid examples: {len(valid_examples)} of {len(df)}")
                self.logger.info(f"Skipped examples: {len(error_examples[lang])}")

                # Log example comparisons from valid examples
                self.logger.info("----- Example translations: -----")
                for count, i in enumerate(valid_examples[:5]):  # Show up to 5 valid examples
                    try:
                        self.logger.info(f"----- Example {count + 1} (index {i}) -----")
                        self.logger.info(f"Source:     {df[source_lang].iloc[i]}")
                        self.logger.info(f"Reference:  {df[lang].iloc[i]}")
                        self.logger.info(f"Prediction: {translations_df[lang].iloc[i]}")
                    except Exception as e:
                        self.logger.error(f"Error displaying example {i}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error calculating metrics for {lang}: {e}")
                metrics[lang] = {
                    "BLEU": 0.0,
                    "chrF": 0.0,
                    "valid_examples": len(valid_examples),
                    "error_examples": len(error_examples[lang]),
                    "error": str(e)
                }

        # Return both metrics and error information
        return metrics, error_examples

    def save_evaluation_report(self, metrics, error_examples, output_dir, test_file):
        """Save evaluation metrics and error information to a report file"""
        report_path = os.path.join(output_dir, "evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write(f"Evaluation Report\n")
            f.write(f"================\n\n")
            f.write(f"Test data: {test_file}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")

            f.write("Metrics Summary:\n")
            for lang, scores in metrics.items():
                f.write(f"\n{lang}:\n")
                for metric, value in scores.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")

            f.write("\n\nError Analysis:\n")
            for lang, errors in error_examples.items():
                if errors:
                    f.write(f"\n{lang} - {len(errors)} problematic examples:\n")
                    for i, error in enumerate(errors[:20]):  # Show up to 20 errors per language
                        f.write(f"\n  Example {i + 1} (index {error['index']}):\n")
                        f.write(f"    Reason: {error['reason']}\n")
                        if 'source' in error:
                            f.write(f"    Source: {error['source']}\n")
                        if 'reference' in error:
                            f.write(f"    Reference: {error['reference']}\n")
                        if 'prediction' in error:
                            f.write(f"    Prediction: {error['prediction']}\n")

                    if len(errors) > 20:
                        f.write(f"\n  ... and {len(errors) - 20} more errors\n")
                else:
                    f.write(f"\n{lang}: No errors encountered\n")

        # Save detailed error information to CSV for further analysis
        error_csv_path = os.path.join(output_dir, "evaluation_errors.csv")
        error_rows = []

        for lang, errors in error_examples.items():
            for error in errors:
                row = {
                    "language": lang,
                    "index": error["index"],
                    "reason": error["reason"]
                }
                if 'source' in error:
                    row["source"] = error["source"]
                if 'reference' in error:
                    row["reference"] = error["reference"]
                if 'prediction' in error:
                    row["prediction"] = error["prediction"]
                error_rows.append(row)

        if error_rows:
            pd.DataFrame(error_rows).to_csv(error_csv_path, index=False)
            self.logger.info(f"Detailed error information saved to {error_csv_path}")

        self.logger.info(f"Evaluation report saved to {report_path}")


def split_data(data_file, val_size=0.15, test_size=0.15, random_seed=42, logger=None):
    """Split data into train, validation, and test sets with shuffling"""
    if logger is None:
        logger = setup_logger()

    logger.info(f"Loading data from {data_file}")

    try:
        df = pd.read_csv(data_file)
        logger.info(f"Successfully loaded {len(df)} examples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

    # Calculate effective test size for the second split
    effective_test_size = test_size / (1 - val_size)

    # First split: train and temp (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=random_seed, shuffle=True
    )

    # Second split: val and test from temp
    val_df, test_df = train_test_split(
        temp_df, test_size=effective_test_size, random_state=random_seed, shuffle=True
    )

    logger.info("Data split complete:")
    logger.info(f"  - Training set:   {len(train_df)} examples ({100 * len(train_df) / len(df):.1f}%)")
    logger.info(f"  - Validation set: {len(val_df)} examples ({100 * len(val_df) / len(df):.1f}%)")
    logger.info(f"  - Test set:       {len(test_df)} examples ({100 * len(test_df) / len(df):.1f}%)")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir, logger=None):
    """Save data splits to files"""
    if logger is None:
        logger = setup_logger()

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Data splits saved to:")
    logger.info(f"  - Training set:   {train_path}")
    logger.info(f"  - Validation set: {val_path}")
    logger.info(f"  - Test set:       {test_path}")

    return train_path, val_path, test_path


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Multilingual Translator Fine-tuning')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['finetune', 'translate', 'batch-translate', 'evaluate', 'pipeline', 'split-data'],
                        help='Mode of operation')
    parser.add_argument('--model', type=str, default="facebook/mbart-large-50-many-to-many-mmt",
                        help='Pretrained model name or path to finetuned model')
    parser.add_argument('--source_lang', type=str, default='en',
                        help='Source language code (e.g., en)')
    parser.add_argument('--target_langs', type=str, default='hi',
                        help='Comma-separated target language codes (e.g., hi,fr)')
    parser.add_argument('--data_file', type=str, help='Path to data CSV file for splitting')
    parser.add_argument('--train_file', type=str, help='Path to training data CSV (if already split)')
    parser.add_argument('--val_file', type=str, help='Path to validation data CSV (if already split)')
    parser.add_argument('--test_file', type=str, help='Path to test data CSV (if already split)')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set size as a fraction of total data')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Test set size as a fraction of total data')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--output_dir', type=str, default='./finetuned_model',
                        help='Directory to save the finetuned model')
    parser.add_argument('--output_file', type=str, help='File to save translations')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of examples to use for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for training')
    parser.add_argument('--text', type=str, help='Text to translate (for translate mode)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (optional)')

    args = parser.parse_args()

    # Create log file path if not provided
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        args.log_file = os.path.join(args.output_dir, f"translator_{timestamp}.log")

    # Set up logger
    logger = setup_logger(args.log_file)

    # Print welcome banner
    logger.info("-" * 70)
    logger.info(" " * 15 + "Multilingual Translator - Training & Inference" + " " * 15)
    logger.info("-" * 70)

    # Process target languages
    target_langs = args.target_langs.split(',')

    # Handle data splitting first if needed
    if args.mode in ['finetune', 'pipeline'] and args.data_file and not all(
            [args.train_file, args.val_file, args.test_file]):
        logger.info("===== DATA SPLITTING =====")
        train_df, val_df, test_df = split_data(
            args.data_file,
            val_size=args.val_size,
            test_size=args.test_size,
            random_seed=args.random_seed,
            logger=logger
        )

        if all([train_df is not None, val_df is not None, test_df is not None]):
            train_path, val_path, test_path = save_splits(train_df, val_df, test_df, args.output_dir, logger)
            args.train_file = train_path
            args.val_file = val_path
            args.test_file = test_path
        else:
            logger.error("Error splitting data. Exiting...")
            return

    # Execute split-data mode and exit
    if args.mode == 'split-data':
        if not args.data_file:
            logger.error("split-data mode requires --data_file")
            parser.error("split-data mode requires --data_file")

        logger.info("===== DATA SPLITTING =====")
        train_df, val_df, test_df = split_data(
            args.data_file,
            val_size=args.val_size,
            test_size=args.test_size,
            random_seed=args.random_seed,
            logger=logger
        )

        if all([train_df is not None, val_df is not None, test_df is not None]):
            save_splits(train_df, val_df, test_df, args.output_dir, logger)
        return

    # Initialize translator
    translator = MultilingualTranslator(model_name=args.model, logger=logger)

    # Execute the requested mode
    if args.mode == 'finetune':
        if not args.train_file or not args.val_file:
            logger.error("finetune mode requires --train_file and --val_file")
            parser.error("finetune mode requires --train_file and --val_file")

        # Load data
        try:
            train_df = pd.read_csv(args.train_file)
            val_df = pd.read_csv(args.val_file)
            logger.info(f"Loaded training data with {len(train_df)} examples")
            logger.info(f"Loaded validation data with {len(val_df)} examples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            parser.error(f"Error loading data: {e}")

        # Fine-tune the model
        model_path = translator.finetune(
            df=train_df,
            val_df=val_df,
            source_lang=args.source_lang,
            target_langs=target_langs,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )
        logger.info(f"Fine-tuned model saved to {model_path}")

    elif args.mode == 'translate':
        if not args.text:
            logger.error("translate mode requires --text")
            parser.error("translate mode requires --text")

        # Translate text
        translations = translator.translate_text(
            text=args.text,
            source_lang=args.source_lang,
            target_langs=target_langs
        )

        # Print translations
        logger.info("Translation Results:")
        logger.info(f"Source ({args.source_lang}): {args.text}")
        for lang, translation in translations.items():
            logger.info(f"Translation ({lang}): {translation}")

    elif args.mode == 'batch-translate':
        if not args.test_file:
            logger.error("batch-translate mode requires --test_file")
            parser.error("batch-translate mode requires --test_file")

        # Load test data
        try:
            test_df = pd.read_csv(args.test_file)
            logger.info(f"Loaded test data with {len(test_df)} examples")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            parser.error(f"Error loading test data: {e}")

        # Generate translations
        translations_df = translator.batch_translate(
            df=test_df,
            source_lang=args.source_lang,
            target_langs=target_langs,
            output_file=args.output_file
        )

    elif args.mode == 'evaluate':
        if not args.test_file:
            logger.error("evaluate mode requires --test_file")
            parser.error("evaluate mode requires --test_file")

        # Determine output file path if not provided
        if not args.output_file:
            args.output_file = os.path.join(args.output_dir, "translations.csv")

        # Load test data
        try:
            test_df = pd.read_csv(args.test_file)
            logger.info(f"Loaded test data with {len(test_df)} examples")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            parser.error(f"Error loading test data: {e}")

        # Generate translations
        translations_df = translator.batch_translate(
            df=test_df,
            source_lang=args.source_lang,
            target_langs=target_langs,
            output_file=args.output_file
        )

        # Evaluate translations
        metrics, error_examples = translator.evaluate_translations(
            df=test_df,
            translations_df=translations_df,
            source_lang=args.source_lang,
            target_langs=target_langs
        )

        # Save evaluation report
        translator.save_evaluation_report(metrics, error_examples, args.output_dir, args.test_file)

    elif args.mode == 'pipeline':
        # Check if we have all necessary data files
        if not all([args.train_file, args.val_file, args.test_file]):
            logger.error("pipeline mode requires --train_file, --val_file, and --test_file")
            parser.error("pipeline mode requires --train_file, --val_file, and --test_file")

        # Load all data
        try:
            train_df = pd.read_csv(args.train_file)
            val_df = pd.read_csv(args.val_file)
            test_df = pd.read_csv(args.test_file)
            logger.info(f"Loaded training data with {len(train_df)} examples")
            logger.info(f"Loaded validation data with {len(val_df)} examples")
            logger.info(f"Loaded test data with {len(test_df)} examples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            parser.error(f"Error loading data: {e}")

        # 1. Fine-tune the model
        logger.info("===== STEP 1: FINE-TUNING =====")
        model_path = translator.finetune(
            df=train_df,
            val_df=val_df,
            source_lang=args.source_lang,
            target_langs=target_langs,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )

        # Reinitialize translator with fine-tuned model
        logger.info("Loading fine-tuned model...")
        translator = MultilingualTranslator(model_name=os.path.join(args.output_dir, "final"), logger=logger)

        # 2. Translate test data
        logger.info("===== STEP 2: BATCH TRANSLATION =====")
        translations_file = os.path.join(args.output_dir, "translations.csv")
        translations_df = translator.batch_translate(
            df=test_df,
            source_lang=args.source_lang,
            target_langs=target_langs,
            output_file=translations_file
        )

        # 3. Evaluate translations
        logger.info("===== STEP 3: EVALUATION =====")
        metrics, error_examples = translator.evaluate_translations(
            df=test_df,
            translations_df=translations_df,
            source_lang=args.source_lang,
            target_langs=target_langs
        )

        # Save evaluation report
        translator.save_evaluation_report(metrics, error_examples, args.output_dir, args.test_file)

        logger.info("===== PIPELINE COMPLETE =====")
        logger.info(f"Fine-tuned model: {os.path.join(args.output_dir, 'final')}")
        logger.info(f"Translations: {translations_file}")
        logger.info(f"Evaluation report: {os.path.join(args.output_dir, 'evaluation_report.txt')}")


if __name__ == "__main__":
    try:
        main()
        logging.getLogger("translator").info("Program completed successfully!")
    except Exception as e:
        logger = logging.getLogger("translator")
        logger.error(f"Error: {e}")
        logger.error("Program terminated with an error.")
        raise