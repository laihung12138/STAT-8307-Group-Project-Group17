import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm.auto import tqdm
from collections import Counter
import string
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
import sys

# Import functions from the utils file
from distilbert_lora_inference_utils import distilbert_lora_prediction

# Check if textstat is installed, if not install it
try:
    import textstat
except ImportError:
    print("Installing textstat...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textstat", "-q"])
    import textstat

from textstat import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    dale_chall_readability_score,
    difficult_words,
)


# Analysis functions
def calculate_complexity(text):
    """Calculate text complexity metrics."""
    try:
        return pd.Series({
            'flesch_ease': flesch_reading_ease(text),
            'flesch_kincaid': flesch_kincaid_grade(text),
            'difficult_words': difficult_words(text),
            'dale_chall': dale_chall_readability_score(text)
        })
    except Exception:
        return pd.Series({
            'flesch_ease': np.nan, 'flesch_kincaid': np.nan,
            'difficult_words': np.nan, 'dale_chall': np.nan
        })

def plot_top_keywords(df, text_column, title_suffix, num_keywords=20, filename_suffix="", output_dir="."):
    """Create a plot of the top keywords in a text corpus."""
    if len(df) == 0:
        print(f"Skipping keyword analysis for '{title_suffix}': No samples.")
        return

    print(f"\nTop {num_keywords} Keywords for {title_suffix}:")
    try:
        vec = CountVectorizer(stop_words='english', max_features=num_keywords).fit(df[text_column])
        bag_of_words = vec.transform(df[text_column])
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        top_df = pd.DataFrame(words_freq[:num_keywords], columns=['Keyword', 'Frequency'])
        print(top_df)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Frequency', y='Keyword', data=top_df, palette='viridis')
        plt.title(f'Top {num_keywords} Keywords ({title_suffix})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"keywords_{filename_suffix}.png"))
        plt.show()

    except ValueError as e:
        print(f"Could not generate keywords for {title_suffix}: {e}") 

def get_top_ngrams(text_series, n=2, num_ngrams=20):
    """Extract the most common n-grams from a series of texts."""
    if len(text_series) == 0:
        return pd.DataFrame(columns=[f'{n}-gram', 'Frequency'])

    # Simple whitespace and punctuation cleaning for n-grams
    def clean_for_ngrams(text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    all_ngrams = []
    cleaned_texts = text_series.apply(clean_for_ngrams)
    for text in cleaned_texts:
        words = text.split()
        if len(words) >= n:
            grams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(grams)

    if not all_ngrams:
        return pd.DataFrame(columns=[f'{n}-gram', 'Frequency'])

    ngram_freq = Counter(all_ngrams)
    common_ngrams = ngram_freq.most_common(num_ngrams)
    return pd.DataFrame(common_ngrams, columns=[f'{n}-gram', 'Frequency'])

def plot_ngrams(df, title, filename, output_dir="."):
    """Plot the top n-grams."""
    if df.empty:
        print(f"Skipping plot for '{title}': No n-grams found.")
        return
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Frequency', y=df.columns[0], data=df, palette='magma')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def run_error_analysis(processed_df, output_dir, use_features=False):
    """
    Run error analysis on the processed data.
    """
    # Define a model-specific prefix for output files
    prefix = f"{'with_features' if use_features else 'no_features'}_"
    
    # Separate by result type
    correct_df = processed_df[processed_df['is_correct']].copy()
    incorrect_df = processed_df[~processed_df['is_correct']].copy()
    tp_df = processed_df[processed_df['result_type'] == 'TP'].copy()
    tn_df = processed_df[processed_df['result_type'] == 'TN'].copy()
    fp_df = processed_df[processed_df['result_type'] == 'FP'].copy()
    fn_df = processed_df[processed_df['result_type'] == 'FN'].copy()

    if len(incorrect_df) == 0:
        print("\nNo misclassifications found on the validation set! Error analysis ends here.")

    # --- TEXT LENGTH ANALYSIS ---
    print("\n--- Analyzing Text Length ---")
    processed_df['text_length'] = processed_df['text'].apply(len)
    correct_df['text_length'] = correct_df['text'].apply(len)
    incorrect_df['text_length'] = incorrect_df['text'].apply(len)
    tn_df['text_length'] = tn_df['text'].apply(len)
    fp_df['text_length'] = fp_df['text'].apply(len)
    tp_df['text_length'] = tp_df['text'].apply(len)
    fn_df['text_length'] = fn_df['text'].apply(len)

    print("\nDescriptive Statistics for Text Length:")
    print("Correct Predictions:\n", correct_df['text_length'].describe())
    print("\nIncorrect Predictions:\n", incorrect_df['text_length'].describe())
    print("\nTN Predictions:\n", tn_df['text_length'].describe())
    print("\nFP Predictions:\n", fp_df['text_length'].describe())
    print("\nTP Predictions:\n", tp_df['text_length'].describe())
    print("\nFN Predictions:\n", fn_df['text_length'].describe())

    # Length histograms and boxplots
    plt.figure(figsize=(12, 6))
    sns.histplot(data=processed_df, x='text_length', hue='is_correct', kde=True, bins=50)
    plt.title('Text Length Distribution (Correct vs. Incorrect Predictions)')
    plt.xlabel('Text Length (Characters)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f"{prefix}length_histogram.png"))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=processed_df, x='is_correct', y='text_length')
    plt.title('Text Length Comparison (Correct vs. Incorrect)')
    plt.xlabel('Prediction Correct?')
    plt.ylabel('Text Length (Characters)')
    plt.xticks([0, 1], ['Incorrect', 'Correct'])
    plt.savefig(os.path.join(output_dir, f"{prefix}length_boxplot.png"))
    plt.show()

    # Length analysis by result type
    type_order = ['TN', 'FP', 'TP', 'FN']
    plot_data = processed_df[processed_df['result_type'].isin(type_order) & processed_df['text_length'].notna()].copy()

    if not plot_data.empty:
        plt.figure(figsize=(13, 7))
        sns.histplot(data=plot_data, x='text_length', hue='result_type', kde=True, bins=50,
                    hue_order=type_order, palette='viridis')
        plt.title('Text Length Distribution by Result Type (TP, TN, FP, FN)')
        plt.xlabel('Text Length (Characters)')
        plt.ylabel('Count')
        plt.legend(title='Result Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}length_histogram_by_type.png"))
        plt.show()

        plt.figure(figsize=(10, 7))
        sns.boxplot(data=plot_data, x='result_type', y='text_length',
                    order=type_order, palette='viridis')
        plt.title('Text Length Comparison by Result Type (TP, TN, FP, FN)')
        plt.xlabel('Result Type')
        plt.ylabel('Text Length (Characters)')
        plt.xticks(rotation=10) 
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}length_boxplot_by_type.png"))
        plt.show()

    # --- TEXT COMPLEXITY ANALYSIS ---
    print("\n--- Analyzing Text Complexity ---")
    tqdm.pandas(desc="Calculating Complexity")
    complexity_scores = processed_df['text'].progress_apply(calculate_complexity)
    processed_df = pd.concat([processed_df, complexity_scores], axis=1)

    # Compare average scores
    print("\nAverage Complexity Scores:")
    print(processed_df.groupby('is_correct')[['difficult_words']].mean())

    # Plot distributions
    complexity_metrics = ['difficult_words']
    for metric in complexity_metrics:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=processed_df, x=metric, hue='is_correct', kde=True, bins=40)
        plt.title(f'{metric.replace("_", " ").title()} Distribution (Correct vs. Incorrect)')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, f"{prefix}complexity_{metric}_hist.png"))
        plt.show()

    # Complexity by result type
    print("\n--- Analyzing Text Complexity by Four Result Types (TN, FP, TP, FN) ---")
    processed_df = processed_df.loc[:, ~processed_df.columns.duplicated(keep='first')]
    plot_data_four_types = processed_df[processed_df['result_type'].isin(type_order)].copy()

    print("\nDescriptive Statistics for Complexity Metrics by Result Type:")
    for metric in complexity_metrics:
        if metric in plot_data_four_types.columns:
            print(f"\n--- {metric.replace('_', ' ').title()} ---")
            stats = plot_data_four_types.groupby('result_type')[metric].describe()
            try:
                print(stats.reindex(type_order))
            except KeyError:
                print("Warning: Not all result types found in data for stats ordering.")
                print(stats) 
        else:
            print(f"Metric '{metric}' not found for descriptive stats.")

    # Plot complexity metrics by result type
    for metric in complexity_metrics:
        if metric in plot_data_four_types and pd.api.types.is_numeric_dtype(plot_data_four_types[metric]):
           
            plot_data_metric = plot_data_four_types.dropna(subset=[metric, 'result_type'])

            if not plot_data_metric.empty:
                # Box Plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=plot_data_metric, x='result_type', y=metric, order=type_order, palette='viridis')
                plt.title(f'{metric.replace("_", " ").title()} Comparison by Result Type')
                plt.xlabel('Result Type')
                plt.ylabel(metric.replace("_", " ").title())
                plt.xticks(rotation=10)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{prefix}complexity_{metric}_boxplot_four_types.png"))
                plt.show()

                # Histogram 
                plt.figure(figsize=(12, 7))
                ax = sns.histplot(data=plot_data_metric, x=metric, hue='result_type',
                                  hue_order=type_order, kde=True, bins=40, palette='viridis',
                                  common_norm=False) 
                plt.title(f'{metric.replace("_", " ").title()} Distribution by Result Type')
                plt.xlabel(metric.replace("_", " ").title())
                plt.ylabel('Count')

                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                     ax.legend(handles=handles, labels=labels, title='Result Type', bbox_to_anchor=(1.02, 1), loc='upper left')
                     plt.subplots_adjust(right=0.85) 
                else:
                     print(f"Could not generate legend for {metric} histogram.")
                     plt.tight_layout() 

                plt.savefig(os.path.join(output_dir, f"{prefix}complexity_{metric}_hist_four_types.png"))
                plt.show()

    # --- KEYWORD ANALYSIS ---
    print("\n--- Analyzing Top Keywords ---")
    plot_top_keywords(correct_df, 'text', 'Correct Predictions', filename_suffix=f"{prefix}correct", output_dir=output_dir)
    plot_top_keywords(incorrect_df, 'text', 'Incorrect Predictions', filename_suffix=f"{prefix}incorrect", output_dir=output_dir)
    plot_top_keywords(tn_df, 'text', 'True Negatives', filename_suffix=f"{prefix}tn", output_dir=output_dir)
    plot_top_keywords(fp_df, 'text', 'False Positives', filename_suffix=f"{prefix}fp", output_dir=output_dir)
    plot_top_keywords(fn_df, 'text', 'False Negatives', filename_suffix=f"{prefix}fn", output_dir=output_dir)
    plot_top_keywords(tp_df, 'text', 'True Positives', filename_suffix=f"{prefix}tp", output_dir=output_dir)

    # --- N-GRAM ANALYSIS ---
    print("\n--- Analyzing N-grams ---")
    for n_gram_size in [2, 3]:
        print(f"\n-- Top {n_gram_size}-grams --")

        # Correct vs Incorrect
        top_ngrams_correct = get_top_ngrams(correct_df['text'], n=n_gram_size, num_ngrams=20)
        print(f"\nTop {n_gram_size}-grams for Correct Predictions:")
        print(top_ngrams_correct)
        plot_ngrams(top_ngrams_correct, f'Top {n_gram_size}-grams (Correct Predictions)', 
                   f"{prefix}ngram{n_gram_size}_correct.png", output_dir=output_dir)

        top_ngrams_incorrect = get_top_ngrams(incorrect_df['text'], n=n_gram_size, num_ngrams=20)
        print(f"\nTop {n_gram_size}-grams for Incorrect Predictions:")
        print(top_ngrams_incorrect)
        plot_ngrams(top_ngrams_incorrect, f'Top {n_gram_size}-grams (Incorrect Predictions)', 
                   f"{prefix}ngram{n_gram_size}_incorrect.png", output_dir=output_dir)

        # By result type
        top_ngrams_tn = get_top_ngrams(tn_df['text'], n=n_gram_size, num_ngrams=15)
        plot_ngrams(top_ngrams_tn, f'Top {n_gram_size}-grams (True Negatives)', 
                   f"{prefix}ngram{n_gram_size}_tn.png", output_dir=output_dir)
        
        top_ngrams_fp = get_top_ngrams(fp_df['text'], n=n_gram_size, num_ngrams=15)
        plot_ngrams(top_ngrams_fp, f'Top {n_gram_size}-grams (False Positives)', 
                   f"{prefix}ngram{n_gram_size}_fp.png", output_dir=output_dir)
        
        top_ngrams_fn = get_top_ngrams(fn_df['text'], n=n_gram_size, num_ngrams=15)
        plot_ngrams(top_ngrams_fn, f'Top {n_gram_size}-grams (False Negatives)', 
                   f"{prefix}ngram{n_gram_size}_fn.png", output_dir=output_dir)
        
        top_ngrams_tp = get_top_ngrams(tp_df['text'], n=n_gram_size, num_ngrams=15)
        plot_ngrams(top_ngrams_tp, f'Top {n_gram_size}-grams (True Positives)', 
                   f"{prefix}ngram{n_gram_size}_tp.png", output_dir=output_dir)

    # --- CONFIDENCE ANALYSIS ---
    print("\n--- Analyzing Prediction Confidence ---")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=processed_df, x='pred_probability', hue='is_correct', kde=True, bins=30)
    plt.title('Prediction Probability Distribution (Correct vs. Incorrect)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f"{prefix}confidence_histogram.png"))
    plt.show()

    print("\nAverage Prediction Probability:")
    print(processed_df.groupby('is_correct')['pred_probability'].mean())

    print("\n--- Analyzing Prediction Confidence by result type ---")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=processed_df, x='pred_probability', hue='result_type', kde=True, bins=30)
    plt.title('Prediction Probability Distribution (TN, FP, FN, TP)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f"{prefix}confidence_histogram_by_type.png"))
    plt.show()

    print("\nAverage Prediction Probability:")
    print(processed_df.groupby('result_type')['pred_probability'].mean())

    # --- High-Confidence Errors Examples Analysis ---
    print("\n--- Analyzing High-Confidence Errors Examples ---")

    # Define the confidence threshold
    confidence_threshold = 0.90 

    # Filter for incorrect predictions with high confidence
    high_confidence_errors_df = processed_df[
        (processed_df['is_correct'] == False) &
        (processed_df['pred_probability'] > confidence_threshold)
    ].copy() 

    # Sort by probability to see the most confident errors first
    high_confidence_errors_df = high_confidence_errors_df.sort_values(by='pred_probability', ascending=False)

    print(f"Found {len(high_confidence_errors_df)} examples where the prediction was incorrect")
    print(f"but the model's confidence in the wrong prediction was > {confidence_threshold:.2f}")

    # Display the top N most confident errors
    num_examples_to_show = 10 

    print(f"\n--- Top {num_examples_to_show} Most Confident Errors ---")

    if len(high_confidence_errors_df) > 0:
        display_cols = ['text', 'label', 'prediction', 'pred_probability']
        
        display_cols = [col for col in display_cols if col in high_confidence_errors_df.columns]

        for index, row in high_confidence_errors_df.head(num_examples_to_show).iterrows():
            print(f"\n--- Example Index: {index} ---")
            print(f"  True Label:       {row['label']}")
            print(f"  Predicted Label:  {row['prediction']}")
            print(f"  Confidence (Wrong): {row['pred_probability']:.4f}")
            # Print the first 500 characters of the text for brevity
            print(f"  Text Snippet:     {row['text'][:500]}...") 
            print("-" * 30)

        # Save high-confidence errors to CSV
        try:
            output_filename = os.path.join(output_dir, f"{prefix}high_confidence_errors_gt_{confidence_threshold:.2f}.csv")
            high_confidence_errors_df.to_csv(output_filename, index=False)
            print(f"\nSaved all {len(high_confidence_errors_df)} high-confidence error examples to: {output_filename}")
        except Exception as e:
            print(f"\nCould not save high-confidence errors to CSV: {e}")
    else:
        print("No high-confidence errors found matching the criteria.")


    # Save the full results dataframe
    processed_df.to_csv(os.path.join(output_dir, f"{prefix}all_analysis_results.csv"), index=False)
    print(f"\nComplete analysis results saved to {output_dir}")
    
    # Return results metrics 
    results = {
        'accuracy': (processed_df['is_correct'].sum() / len(processed_df)),
        'tn': len(tn_df),
        'fp': len(fp_df),
        'fn': len(fn_df),
        'tp': len(tp_df)
    }
    
    return results


def main():
    """Main function to parse arguments and run the error analysis."""
    parser = argparse.ArgumentParser(description='Run error analysis on various text classification models')
    parser.add_argument('--validation_data', type=str, default='data/text_validation.csv',
                        help='Path to validation data CSV file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model artifacts for model')
    parser.add_argument('--use_features', type=str, default='False',
                        help='Whether to use linguistic features (True/False)')
    parser.add_argument('--output_dir', type=str, default='error_analysis_output',
                        help='Directory to save output files')
    
    args = parser.parse_args()

    # Convert string to boolean
    use_features = args.use_features.lower() == 'true'

    # Extract model name from the path
    model_name = "unknown_model"
    if args.model:
        model_name = os.path.basename(args.model.rstrip('/'))   

    # Add feature indicator to model name
    feature_indicator = "with_features" if use_features else "no_features"

    # Create a unique output directory for each run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{model_name}_{feature_indicator}_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print(f"\n{'='*80}\nPreparing data for model")
    processed_df = distilbert_lora_prediction(
        input_path=args.validation_data,
        model_artifacts_path=args.model,
        use_features=use_features
    )
            
    print(f"\n{'='*80}\nRunning error analysis for model\n{'='*80}")
    results['model_results'] = run_error_analysis(
        processed_df=processed_df,
        output_dir=output_dir,
        use_features=use_features
    )
    
    print(f"\nAll error analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

 
### python error_analysis.py --model finetuned_model/distilbert_lora_no_features_finetuned_detector --use_features False
### python error_analysis.py --model finetuned_model/distilbert_lora_with_features_finetuned_detector --use_features true
