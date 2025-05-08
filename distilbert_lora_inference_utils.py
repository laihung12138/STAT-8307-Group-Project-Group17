import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import re
import nltk
import sys
import joblib
from tqdm.auto import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from peft import PeftModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Import distilbert custom model and preprocessing functions
from models import DistilBertWithFeatures
from data_preprocessing import extract_linguistic_features_dis_lora

# Download stopwords if not already present
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)  
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print(f"Downloading NLTK stopwords to {nltk_data_dir}...")
    try:
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"Error downloading stopwords: {e}")
        print("Attempting alternative download method...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "nltk.downloader", "-d", nltk_data_dir, "stopwords"])
    print("Download complete.")


class TextFeatureDataset(TorchDataset):
    """Dataset class that handles text with additional features."""
    def __init__(self, encodings, features=None, labels=None):
        self.encodings = encodings
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.features is not None:
            item['features'] = self.features[idx]
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) if not isinstance(self.labels[idx], torch.Tensor) else self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def extract_and_scale_features(df, scaler_path="processed_distilbert_data/feature_scaler.joblib"):
    """Extract and scale linguistic features from text data using saved scaler."""
    print("\nExtracting linguistic features...")
    tqdm.pandas(desc="Extracting Features")
    linguistic_features_list = df['text'].progress_apply(extract_linguistic_features_dis_lora)
    features_df = pd.DataFrame(list(linguistic_features_list), index=df.index) 
    feature_names = list(features_df.columns)
    NUM_LINGUISTIC_FEATURES = len(feature_names)
    print(f"Extracted {NUM_LINGUISTIC_FEATURES} features: {feature_names}")

    # Load the pre-fitted scaler
    try:
        print(f"\nLoading pre-fitted scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Could not load pre-fitted scaler: {e}")
        print("Creating and fitting a new scaler...")
        scaler = MinMaxScaler()
        scaler.fit(features_df)
    
    # Scale the features using the loaded/created scaler
    print("Scaling features...")
    scaled_features = scaler.transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)
    scaled_features_tensor = torch.tensor(scaled_features_df.values, dtype=torch.float32)
    
    return scaled_features_tensor, NUM_LINGUISTIC_FEATURES


def load_data_from_csv(data_path):
    """Load and preprocess data from CSV."""
    print(f"\nLoading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        # Handle potential NaN/missing text
        if 'text' in df.columns:
            df['text'] = df['text'].fillna('').astype(str)
            print(f"Loaded {len(df)} samples.")
            
            # Report label distribution 
            if 'label' in df.columns:
                print("Label distribution:")
                print(df['label'].value_counts())
            return df
        else:
            raise ValueError("CSV must contain a 'text' column.")
    except FileNotFoundError:
        raise SystemExit(f"Error: Data file not found at {data_path}")
    except Exception as e:
        raise SystemExit(f"Error loading data: {e}")
    

def load_model_and_tokenizer(model_artifacts_path='finetuned_model/distilbert_lora_no_features_finetuned_detector', 
                             use_features=False, base_model_name='distilbert-base-uncased', num_linguistic_features=0):
    """Load the appropriate model and tokenizer."""
    print(f"\nLoading model and tokenizer from: {model_artifacts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_artifacts_path)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        # Try from base model as fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            print(f"Tokenizer loaded from base model: {base_model_name}")
        except Exception as e2:
            raise SystemExit(f"Error loading tokenizer: {e} and {e2}")

    # Load the appropriate model
    try:
        adapter_path = os.path.join(model_artifacts_path, "final_lora_adapter")
        print(f"Attempting to load LoRA adapter from: {adapter_path}")
        
        if not os.path.isdir(adapter_path):
             raise FileNotFoundError(f"Adapter directory not found at '{adapter_path}'.")
        
        if use_features:
            # Load custom model with features
            config = AutoConfig.from_pretrained(base_model_name, num_labels=2)
            base_model = DistilBertWithFeatures.from_pretrained(
                base_model_name,
                config=config,
                num_linguistic_features=num_linguistic_features,
            )
            print(f"Custom base model 'DistilBertWithFeatures' loaded with {num_linguistic_features} features.")
        else:
            # Load standard model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2,
            )
            print(f"Standard base model '{base_model_name}' loaded.")
        
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        print("LoRA adapter loaded successfully.")
        
        return model, tokenizer, device

    except FileNotFoundError as e:
         raise SystemExit(f"Error loading PEFT model: {e}")
    except Exception as e:
         import traceback
         traceback.print_exc()
         raise SystemExit(f"Error loading model: {e}")


def run_inference(model, df, tokenizer, device, scaled_features_tensor=None, use_features=False, max_length=512, batch_size=16):
    """Run inference on data and return results."""
    print("\nPreprocessing validation data for prediction...")
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True, padding=True, max_length=max_length, return_tensors='pt'
    )
    print("Tokenization complete.")

    # Create the appropriate dataset and dataloader based on model type
    if use_features:
        val_dataset = TextFeatureDataset(encodings, features=scaled_features_tensor, labels=df['label'].values)
    else:
        val_dataset = TextFeatureDataset(encodings, features=None, labels=df['label'].values)
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("DataLoader created.")

    # Run inference
    print("\nRunning inference...")
    predicted_labels = []
    predicted_probabilities = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Pass features if model uses them
            if use_features:
                features = batch['features'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            predicted_labels.extend(predictions.cpu().numpy())
            predicted_probabilities.extend(probabilities.cpu().numpy())

    print("Predictions finished.") 
    
    # Process results
    result_df = df.copy()
    result_df['prediction'] = predicted_labels
    result_df['pred_probability'] = [p[1] for p in predicted_probabilities]  # Probability of positive class
    
    # Calculate accuracy metrics if true label exsit
    if 'label' in result_df.columns:
        result_df['is_correct'] = (result_df['label'] == result_df['prediction'])
        
        # Identify TP, TN, FP, FN
        result_df['result_type'] = 'NA'
        result_df.loc[(result_df['is_correct'] == True) & (result_df['label'] == 0), 'result_type'] = 'TN' 
        result_df.loc[(result_df['is_correct'] == True) & (result_df['label'] == 1), 'result_type'] = 'TP'
        result_df.loc[(result_df['is_correct'] == False) & (result_df['label'] == 0), 'result_type'] = 'FP'
        result_df.loc[(result_df['is_correct'] == False) & (result_df['label'] == 1), 'result_type'] = 'FN'

        print("Prediction results combined. Counts:")
        print(result_df['result_type'].value_counts())
        
        # Calculate and print metrics
        accuracy = result_df['is_correct'].mean()
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(result_df['label'], result_df['prediction']))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(result_df['label'], result_df['prediction'])
        print(cm)
    else:
        print(f"Predictions complete. {len(result_df)} samples processed.")
        print("Prediction distribution:")
        print(result_df['prediction'].value_counts())
    
    return result_df


def distilbert_lora_prediction(input_path='data/text_validation.csv', 
                               model_artifacts_path='finetuned_model/distilbert_lora_no_features_finetuned_detector',
                                use_features=False, base_model_name="distilbert-base-uncased", 
                                scaler_path="processed_distilbert_data/feature_scaler.joblib"):
    """
    Process a file of texts for prediction.
    """

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df = load_data_from_csv(input_path)
    
    # Process features if needed
    if use_features:
        scaled_features_tensor, num_linguistic_features = extract_and_scale_features(df, scaler_path)
    else:
        scaled_features_tensor, num_linguistic_features = None, 0
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        model_artifacts_path, 
        use_features, 
        base_model_name, 
        num_linguistic_features
    )
    
    # Run inference
    processed_df = run_inference(
        model, 
        df, 
        tokenizer, 
        device, 
        scaled_features_tensor, 
        use_features
    )
    
    return processed_df



if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='DistilBERT-LoRA Text Classification Utility')
    
    # Add arguments for the predict functionality
    parser.add_argument('--input', type=str, required=True, 
                      help='Path to input CSV file with texts (must have "text" column)')
    parser.add_argument('--output', type=str, 
                      help='Path to save predictions (default: input_predictions.csv)')
    parser.add_argument('--model', type=str, 
                      help='Path to model artifacts directory')
    parser.add_argument('--use_features', action='store_true', 
                      help='Use linguistic features with the model')
    parser.add_argument('--base_model', type=str, default='distilbert-base-uncased',
                      help='Base model name (default: distilbert-base-uncased)')
    parser.add_argument('--scaler', type=str, default='processed_distilbert_data/feature_scaler.joblib',
                      help='Path to feature scaler joblib file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine model path based on whether using features
    if args.model:
        model_path = args.model
    else:
        if args.use_features:
            model_path = "finetuned_model/distilbert_lora_with_features_finetuned_detector"
        else:
            model_path = "finetuned_model/distilbert_lora_no_features_finetuned_detector"
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_predictions.csv"
    
    print(f"Running prediction on file: {args.input}")
    print(f"Using model: {model_path}")
    print(f"Using features: {'Yes' if args.use_features else 'No'}")
    
    # Run prediction
    results_df = distilbert_lora_prediction(
        input_path=args.input,
        model_artifacts_path=model_path,
        use_features=args.use_features,
        base_model_name=args.base_model,
        scaler_path=args.scaler
    )
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    # Print summary
    if 'label' in results_df.columns:
        accuracy = (results_df['label'] == results_df['prediction']).mean()
        print(f"\nSummary:")
        print(f"Total samples: {len(results_df)}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Interpret predictions
        true_negative = ((results_df['label'] == 0) & (results_df['prediction'] == 0)).sum()
        true_positive = ((results_df['label'] == 1) & (results_df['prediction'] == 1)).sum()
        false_negative = ((results_df['label'] == 1) & (results_df['prediction'] == 0)).sum()
        false_positive = ((results_df['label'] == 0) & (results_df['prediction'] == 1)).sum()
        
        print(f"True Negatives: {true_negative}")
        print(f"True Positives: {true_positive}")
        print(f"False Negatives: {false_negative}")
        print(f"False Positives: {false_positive}")
    else:
        human_count = (results_df['prediction'] == 0).sum()
        ai_count = (results_df['prediction'] == 1).sum()
        print(f"\nSummary:")
        print(f"Total samples: {len(results_df)}")
        print(f"Classified as Human: {human_count} ({human_count/len(results_df)*100:.1f}%)")
        print(f"Classified as AI: {ai_count} ({ai_count/len(results_df)*100:.1f}%)")