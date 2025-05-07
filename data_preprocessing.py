import numpy as np
import pandas as pd
import json
import re
import textstat
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import joblib

def initialize_data():
    # Check if "data_merged_original.csv" exists in the data
    try:
        data = pd.read_csv("./data/data_merged_original_yuke.csv")
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("File not found. Merging data...")

        # Define a dataframe for the data source
        data = pd.DataFrame(columns=['text', 'label'])

        # Read the data source from json file
        data_source = json.load(open("./data_source.json", "r"))
        for dataset_name, config in data_source.items():
            print(f"Processing {dataset_name}...")

            # Read the dataset
            dataset = pd.read_csv("./data/"+config["path"])

            # Keep only "text_column" and "label_column"
            dataset = dataset[[config["text_column"], config["label_column"]]]
            dataset.columns = ['text', 'label']

            # Add the dataset to the main dataframe
            data = pd.concat([data, dataset], ignore_index=True)

        # Save the merged data to a CSV file
        data.to_csv("./data/data_merged_original.csv", index=False)
        print("Data merged and saved to data_merged_original.csv.")


    return data

def clean_text(text):
    """Clean the text data by removing special characters and converting to lowercase."""
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

def extract_linguistic_features(text):
    """Extract linguistic features from the text data."""
    features = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'smog_index': textstat.smog_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'difficult_words': textstat.difficult_words(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'gunning_fog': textstat.gunning_fog(text),
        'text_standard': textstat.text_standard(text, float_output=True)
    }
    return features

def minmax_scale_features(features_df):
    """Scale the features using Min-Max scaling."""
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    return pd.DataFrame(scaled_features, columns=features_df.columns)

def tokenize_and_pad(texts, max_words, max_length):
    """Tokenize and pad the text data."""
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences, tokenizer

def preprocess_data(max_words=10000, max_length=100, features_addition=True):
    # Initialize data
    data = initialize_data()

    # Check if text column contains duplicates
    if data['text'].duplicated().any():
        print("Duplicate text entries found. Removing duplicates...")
        data = data.drop_duplicates(subset=['text'])
    else:
        print("No duplicate text entries found.")

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)

    if features_addition == True:
        print("Linguistic features extraction is enabled.")
        # Add linguistic features to the data
        features = data['text'].apply(extract_linguistic_features)
        features_df = pd.DataFrame(features.tolist())

        print("Linguistic features extracted.")
        print(features_df.head())

        # Scale the features using Min-Max scaling
        scaled_features_df = minmax_scale_features(features_df)
        print("Features scaled using Min-Max scaling.")
    else:
        print("Linguistic features extraction is disabled.")

    # Tokenize
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    print("Text data tokenized and padded.")

    # Extract labels
    labels = data['label'].values
    print("Labels extracted.")

    if features_addition == True:
        print("Combined word-level and linguistic features.")
        # Split the data into training and testing sets
        X_train_word, X_test_word, X_train_linguistic, X_test_linguistic, y_train, y_test = train_test_split(
            padded_sequences, scaled_features_df.values, labels, test_size=0.2, random_state=42
        )
        print("Data split into training and testing sets.")

        X_train_word, X_val_word, X_train_linguistic, X_val_linguistic, y_train, y_val = train_test_split(
        X_train_word, X_train_linguistic, y_train, test_size=0.2, random_state=42
        )
        print("Training set further split into training and validation sets.")

        # Save the validation set to 1 single CSV file: x_val_word, x_val_linguistic, y_val
        val_df = pd.DataFrame(X_val_word, columns=[f'word_{i}' for i in range(X_val_word.shape[1])])
        val_df_linguistic = pd.DataFrame(X_val_linguistic, columns=scaled_features_df.columns)
        val_df['label'] = y_val
        val_df_linguistic['label'] = y_val
        val_df.to_csv("./data/text_validation.csv", index=False)
        val_df_linguistic.to_csv("./data/val_data_linguistic.csv", index=False)

        return (
        X_train_word, X_test_word, X_val_word,
        X_train_linguistic, X_test_linguistic, X_val_linguistic,
        y_train, y_test, y_val,
        tokenizer
        )
    else:
        # Split the data into training and testing sets
        X_train_word, X_test_word, y_train, y_test = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )
        print("Data split into training and testing sets.")

        X_train_word, X_val_word, y_train, y_val = train_test_split(
            X_train_word, y_train, test_size=0.2, random_state=42
        )
        print("Training set further split into training and validation sets.")

        # Save the validation set to 1 single CSV file: x_val_word, y_val
        val_df = pd.DataFrame(X_val_word, columns=[f'word_{i}' for i in range(X_val_word.shape[1])])
        val_df['label'] = y_val
        val_df.to_csv("./data/text_validation.csv", index=False)

        return (
            X_train_word, X_test_word, X_val_word,
            y_train, y_test, y_val,
            tokenizer
        )

# ====================================================================================================================================================
# ====================================================================================================================================================

# Distributed LoRA
def initialize_data_dis_lora():
    data = pd.read_csv("./data/data_merged_original_yuke.csv")
    print("Data loaded successfully.")
    return data

def clean_text_dis_lora(text):
    """Clean the text data by removing special characters and converting to lowercase."""
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

def preprocess_data_dis_lora():
    # Initialize data
    data = initialize_data_dis_lora()

    # Check if text column contains duplicates
    if data['text'].duplicated().any():
        print("Duplicate text entries found. Removing duplicates...")
        data = data.drop_duplicates(subset=['text'])
    else:
        print("No duplicate text entries found.")

    # Clean the text data
    data['text'] = data['text'].apply(clean_text_dis_lora)

    # --- Split the Combined DataFrame ---
    print("\nSplitting combined DataFrame into training/testing sets (80/20)...")
    train_val_df, test_df = train_test_split(
        data,
        test_size=0.2,             
        random_state=42)

    print(f"Test set DataFrame size: {len(test_df)}")

    print("\nSplitting training/validation set further into final training and validation sets (80/20)...")
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42
        )

    print(f"Final training set DataFrame size: {len(train_df)}")
    print(f"Final validation set DataFrame size: {len(val_df)}")
    print("Splitting complete.")

    # Check label distribution 
    print("\n=== Label Distribution Analysis ===")
    
    # Function to analyze and print label distribution
    def label_dist(df, name):
        label_counts = df['label'].value_counts().sort_index()
        label_percentages = df['label'].value_counts(normalize=True).sort_index() * 100
        
        print(f"\n{name} Set Label Distribution:")
        print(f"Total samples: {len(df)}")
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
    
    # Analyze each dataset
    label_dist(data, "Full")
    label_dist(train_df, "Training")
    label_dist(val_df, "Validation")
    label_dist(test_df, "Test")
    
    return train_df, val_df, test_df

def split_raw_data_to_csv_dis_lora():
    print("Starting data preprocessing and splitting for CSV saving...")
    try:
        # Define output directory for CSV files
        csv_output_dir = "./data"

        # Get the split DataFrames
        train_df, val_df, test_df = preprocess_data_dis_lora()

        print("\nPreprocessing and splitting finished successfully.")

        # --- Save the processed data as CSV ---
        print(f"\nSaving processed data as CSV files to directory: {csv_output_dir}")
        os.makedirs(csv_output_dir, exist_ok=True) # Create dir if it doesn't exist

        # Define file paths
        train_csv_path = os.path.join(csv_output_dir, "text_train.csv")
        val_csv_path = os.path.join(csv_output_dir, "text_validation.csv")
        test_csv_path = os.path.join(csv_output_dir, "text_test.csv")

        # Save each DataFrame to CSV
        train_df.to_csv(train_csv_path, index=False)
        print(f"Saved: {train_csv_path} (shape: {train_df.shape})")

        val_df.to_csv(val_csv_path, index=False)
        print(f"Saved: {val_csv_path} (shape: {val_df.shape})")

        test_df.to_csv(test_csv_path, index=False)
        print(f"Saved: {test_csv_path} (shape: {test_df.shape})")

        print("\nCSV Data saving complete.")

    except Exception as e:
        print(f"\nAn error occurred during preprocessing or saving: {e}")
        import traceback
        traceback.print_exc()

def extract_linguistic_features_dis_lora(text):
    """Extract linguistic features from the text data."""
    if not isinstance(text, str): text = str(text)
    try:
        features = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
            'difficult_words': textstat.difficult_words(text),
            'linsear_write_formula': textstat.linsear_write_formula(text),
            'gunning_fog': textstat.gunning_fog(text),
            'text_standard': textstat.text_standard(text, float_output=True)
        }
    except Exception: 
         features = {
            'flesch_reading_ease': np.nan, 'flesch_kincaid_grade': np.nan,
            'smog_index': np.nan, 'dale_chall_readability_score': np.nan,
            'difficult_words': np.nan, 'linsear_write_formula': np.nan,
            'gunning_fog': np.nan, 'text_standard': np.nan
        }
    return features

# Model
MODEL_NAME_DIS_LORA = "distilbert-base-uncased"
MAX_LENGTH_DIS_LORA = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DIS_LORA)
def tokenize_functions_dis_lora(examples):
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH_DIS_LORA,
        return_tensors='np'
    )

def preprocess_data_for_distilberts_dis_lora(train_csv_path, val_csv_path, test_csv_path):
    print("Starting DistilBERT preprocessing function...")

    # Load Split Data 
    try:
        print(f"Loading data...")
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)
        print("Train, validation, and test CSV files loaded.")

        # Basic validation
        for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
             if 'text' not in df.columns or 'label' not in df.columns:
                  raise ValueError(f"Missing 'text' or 'label' column in {df_name} CSV.")
             df['label'] = df['label'].astype(int) # Ensure labels are integers
             df['text'] = df['text'].astype(str)   # Ensure text is string

    except FileNotFoundError as e:
        print(f"Error: CSV file not found. Make sure '{e.filename}' exists.")
        raise 
    except Exception as e:
        print(f"An error occurred loading CSV files: {e}")
        raise 

    # Feature Extraction
    print("Extracting linguistic features...")
    train_features_list = train_df['text'].apply(extract_linguistic_features).tolist()
    val_features_list = val_df['text'].apply(extract_linguistic_features).tolist()
    test_features_list = test_df['text'].apply(extract_linguistic_features).tolist()

    train_features_df = pd.DataFrame(train_features_list, index=train_df.index)
    val_features_df = pd.DataFrame(val_features_list, index=val_df.index)
    test_features_df = pd.DataFrame(test_features_list, index=test_df.index)
    print("Feature extraction complete.")

    # Feature Scaling (Fit on Train, Transform All)
    print("Scaling linguistic features...")
    scaler = MinMaxScaler()
    print("Fitting MinMaxScaler on training features...")
    scaler.fit(train_features_df)
    print("Scaler fitted.")

    print("Transforming features for train, validation, and test sets...")
    scaled_train_features = scaler.transform(train_features_df)
    scaled_val_features = scaler.transform(val_features_df)
    scaled_test_features = scaler.transform(test_features_df)
    print("Feature scaling complete.")

    # Tokenization
    print("Tokenizing text...")

    train_texts_dict = {'text': train_df['text'].tolist()}
    val_texts_dict = {'text': val_df['text'].tolist()}
    test_texts_dict = {'text': test_df['text'].tolist()}

    train_tokenized = tokenize_functions_dis_lora(train_texts_dict)
    val_tokenized = tokenize_functions_dis_lora(val_texts_dict)
    test_tokenized = tokenize_functions_dis_lora(test_texts_dict)
    print("Tokenization complete.")

    # Extract Labels 
    print("Extracting labels...")
    train_labels = train_df['label'].values
    val_labels = val_df['label'].values
    test_labels = test_df['label'].values
    print("Labels extracted.")

    # Prepare Output Tuples
    train_data = (
        train_tokenized['input_ids'],
        train_tokenized['attention_mask'],
        scaled_train_features,
        train_labels
    )
    val_data = (
        val_tokenized['input_ids'],
        val_tokenized['attention_mask'],
        scaled_val_features,
        val_labels
    )
    test_data = (
        test_tokenized['input_ids'],
        test_tokenized['attention_mask'],
        scaled_test_features,
        test_labels
    )

    print("Preprocessing function finished successfully.")
    return train_data, val_data, test_data, scaler

def data_preprocessing_original_file_dis_lora():
    ### Input Paths
    split_data_dir = "./data"
    train_csv = os.path.join(split_data_dir, "text_train.csv")
    val_csv = os.path.join(split_data_dir, "text_validation.csv")
    test_csv = os.path.join(split_data_dir, "text_test.csv")

    ### Output Paths 
    output_npy_dir = "./processed_distilbert_data"
    scaler_path = os.path.join(output_npy_dir, "feature_scaler.joblib")
    tokenizer_save_path = os.path.join(output_npy_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DIS_LORA)
    
    # Call the Preprocessing Function 
    try:
        train_data, val_data, test_data, fitted_features_scaler = preprocess_data_for_distilberts_dis_lora(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            test_csv_path=test_csv
        )

        # Unpack the returned data
        train_ids, train_mask, train_features, train_labels = train_data
        val_ids, val_mask, val_features, val_labels = val_data
        test_ids, test_mask, test_features, test_labels = test_data

        print("\nData returned from function:")
        print(f"  Train IDs shape: {train_ids.shape}")
        print(f"  Val Features shape: {val_features.shape}")
        print(f"  Test Labels shape: {test_labels.shape}")
        print(f"  Scaler type: {type(fitted_features_scaler)}")

        # Save the Processed Data and Artifacts
        os.makedirs(output_npy_dir, exist_ok=True)

        # Define file paths and data arrays to save
        save_items = {
            'train_ids.npy': train_ids, 'train_mask.npy': train_mask,
            'train_features.npy': train_features, 'train_labels.npy': train_labels,
            'val_ids.npy': val_ids, 'val_mask.npy': val_mask,
            'val_features.npy': val_features, 'val_labels.npy': val_labels,
            'test_ids.npy': test_ids, 'test_mask.npy': test_mask,
            'test_features.npy': test_features, 'test_labels.npy': test_labels,
        }

        # Save each NumPy array
        for filename, data_array in save_items.items():
            filepath = os.path.join(output_npy_dir, filename)
            np.save(filepath, data_array)
            print(f"  Saved: {filepath} (shape: {data_array.shape})")

        # Save the fitted scaler
        joblib.dump(fitted_features_scaler, scaler_path)
        print(f"  Saved scaler to: {scaler_path}")

        # Save the tokenizer
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"  Saved tokenizer to: {tokenizer_save_path}")

    except Exception as e:
        print(f"\nAn error occurred during the preprocessing function call or saving: {e}")
        import traceback
        traceback.print_exc()
    

# ====================================================================================================================================================
# ====================================================================================================================================================
def tokenize_and_pad_freeze_last_layer(data, max_length=128):
    """
    Tokenize and pad text data using a pretrained tokenizer.
    Optimized for reduced memory usage.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Convert data['text'] to a Python list if necessary
    texts = data['text'].tolist()  # Assuming data['text'] is a Pandas Series
    
    # Tokenize and pad with truncation
    sequences = tokenizer(
        texts,
        padding=True,  # Add padding to the longest sequence
        truncation=True,  # Truncate sequences exceeding max_length
        max_length=max_length,  # Set maximum sequence length
        return_tensors="pt"  # Return PyTorch tensors
    )
    
    # Convert tensors to half precision to save memory
    sequences = {key: value.to(torch.float16) if torch.is_floating_point(value) else value for key, value in sequences.items()}
    
    return sequences

class MultimodalDataset_nofeature(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx]).long()
        }
    
    def __len__(self):
        return len(self.labels)
    
class MultimodalDataset_feature(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, linguistic_features, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.linguistic_features = linguistic_features
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "features": self.linguistic_features[idx],
            "labels": torch.tensor(self.labels[idx]).long()
        }
    
    def __len__(self):
        return len(self.labels)

def data_preprocessing_freeze_last_layer_nofeature(data, max_length=128):
    if data['text'].duplicated().any():
        print("Duplicate text entries found. Removing duplicates...")
        data = data.drop_duplicates(subset=['text'])
    else:
        print("No duplicate text entries found.")

    print("Data after removing duplicates:")
    print(data.info())
    # print(data.describe())

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)
    print("Text data cleaned.")
    
    # Tokenize and pad
    sequences = tokenize_and_pad_freeze_last_layer(data, max_length=max_length)
    input_ids = sequences["input_ids"]
    attention_mask = sequences["attention_mask"]
    labels = data['label'].values
    texts = data['text'].values

    assert len(input_ids) == len(attention_mask) == len(labels), "样本数不一致"

    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    def split_by_indices(data, indices):
        if isinstance(data, torch.Tensor):
            return data[indices]
        else:
            return data.iloc[indices]  

    X_train = (
        input_ids[train_indices],
        attention_mask[train_indices]
    )
    X_val = (
        input_ids[val_indices],
        attention_mask[val_indices]
    )
    X_test = (
        input_ids[test_indices],
        attention_mask[test_indices]
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    datas_train = texts[train_indices]
    datas_val = texts[val_indices]
    datas_test = texts[test_indices]

    input_ids_train, attention_mask_train = X_train
    input_ids_val, attention_mask_val = X_val
    input_ids_test, attention_mask_test = X_test

    train_dataset = MultimodalDataset_nofeature(
        input_ids_train, attention_mask_train, y_train
    )
    val_dataset = MultimodalDataset_nofeature(
        input_ids_val, attention_mask_val, y_val
    )
    test_dataset = MultimodalDataset_nofeature(
        input_ids_test, attention_mask_test, y_test
    )

    return train_dataset, val_dataset, test_dataset, y_train, y_val, y_test, datas_train, datas_val, datas_test


def data_preprocessing_freeze_last_layer_feature(data, max_length=128):
    if data['text'].duplicated().any():
        print("Duplicate text entries found. Removing duplicates...")
        data = data.drop_duplicates(subset=['text'])
    else:
        print("No duplicate text entries found.")

    print("Data after removing duplicates:")
    print(data.info())
    # print(data.describe())

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)
    print("Text data cleaned.")
    
    # Add linguistic features to the data
    features = data['text'].apply(extract_linguistic_features)
    features_df = pd.DataFrame(features.tolist())

    print("Linguistic features extracted.")
    print(features_df.head())
    
    scaled_features_df = minmax_scale_features(features_df)
    print("Features scaled using Min-Max scaling.")

    
    # Tokenize and pad
    sequences = tokenize_and_pad_freeze_last_layer(data, max_length=max_length)
    input_ids = sequences["input_ids"]
    attention_mask = sequences["attention_mask"]
    linguistic_features = torch.tensor(scaled_features_df.values, dtype=torch.float16)
    labels = data['label'].values
    texts = data['text'].values

    assert len(input_ids) == len(attention_mask) == len(labels)== len(linguistic_features), "样本数不一致"

    # 使用索引统一分割
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    def split_by_indices(data, indices):
        if isinstance(data, torch.Tensor):
            return data[indices]
        else:
            return data.iloc[indices]  

    X_train = (
        input_ids[train_indices],
        attention_mask[train_indices],
        linguistic_features[train_indices]
    )
    X_val = (
        input_ids[val_indices],
        attention_mask[val_indices],
        linguistic_features[val_indices]
    )
    X_test = (
        input_ids[test_indices],
        attention_mask[test_indices],
        linguistic_features[test_indices]
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    datas_train = texts[train_indices]
    datas_val = texts[val_indices]
    datas_test = texts[test_indices]

    input_ids_train, attention_mask_train, linguistic_train = X_train
    input_ids_val, attention_mask_val, linguistic_val = X_val
    input_ids_test, attention_mask_test, linguistic_test  = X_test

    train_dataset = MultimodalDataset_feature(
        input_ids_train, attention_mask_train,linguistic_train, y_train
    )
    val_dataset = MultimodalDataset_feature(
        input_ids_val, attention_mask_val, linguistic_val, y_val
    )
    test_dataset = MultimodalDataset_feature(
        input_ids_test, attention_mask_test, linguistic_test, y_test
    )

    return train_dataset, val_dataset, test_dataset, y_train, y_val, y_test, datas_train, datas_val, datas_test

# ====================================================================================================================================================
# ====================================================================================================================================================

def extract_linguistic_features_sgd(text):
    """Extract linguistic features from text."""
    try:
        features = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_readability': textstat.dale_chall_readability_score(text),
            'gunning_fog': textstat.gunning_fog(text),
            'difficult_words': textstat.difficult_words(text),
            'lexicon_count': textstat.lexicon_count(text),
            'sentence_count': textstat.sentence_count(text),
            'char_count': textstat.char_count(text),
            'letter_count': textstat.letter_count(text),
            'word_count': len(text.split()),
            'unique_word_ratio': len(set(text.split())) / max(1, len(text.split())),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'punctuation_count': sum(1 for c in text if c in '.,;:!?\''),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text))
        }
    except:
        features = {k: 0 for k in [
            'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index',
            'dale_chall_readability', 'gunning_fog', 'difficult_words',
            'lexicon_count', 'sentence_count', 'char_count', 'letter_count',
            'word_count', 'unique_word_ratio', 'avg_word_length',
            'punctuation_count', 'uppercase_ratio'
        ]}
    return features

def load_and_clean_data_sgd(train_csv_path="./data/text_train.csv",
                       val_csv_path="./data/text_validation.csv",
                       test_csv_path="./data/text_test.csv"):
    """Load and clean the dataset with default paths."""
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Data cleaning
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=['text', 'label'], inplace=True)
        df['label'] = df['label'].astype(int)
        df['text'] = df['text'].astype(str)

    return train_df, val_df, test_df

def preprocess_data_sgd(use_linguistic_features=False,
                   train_csv_path="./data/text_train.csv",
                   val_csv_path="./data/text_validation.csv",
                   test_csv_path="./data/text_test.csv"):
    """Main preprocessing function with default paths."""
    train_df, val_df, test_df = load_and_clean_data_sgd(
        train_csv_path, val_csv_path, test_csv_path
    )
    
    # Text vectorization
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    X_train_text = vectorizer.fit_transform(train_df['text'])
    X_val_text = vectorizer.transform(val_df['text'])
    X_test_text = vectorizer.transform(test_df['text'])
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join("processed_data", "sgd_vectorizer.joblib"))
    
    if use_linguistic_features:
        # Extract and scale linguistic features
        train_features = train_df['text'].apply(extract_linguistic_features_sgd).apply(pd.Series)
        val_features = val_df['text'].apply(extract_linguistic_features_sgd).apply(pd.Series)
        test_features = test_df['text'].apply(extract_linguistic_features_sgd).apply(pd.Series)
        
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)
        
        # Convert sparse text features to dense for SGD
        X_train_text = X_train_text.toarray()
        X_val_text = X_val_text.toarray()
        X_test_text = X_test_text.toarray()
        
        # Combine features
        X_train = np.hstack([X_train_text, train_features])
        X_val = np.hstack([X_val_text, val_features])
        X_test = np.hstack([X_test_text, test_features])
        
        # Save scaler
        joblib.dump(scaler, os.path.join("processed_data", "sgd_scaler.joblib"))
    else:
        # Convert sparse to dense for SGD
        X_train = X_train_text.toarray()
        X_val = X_val_text.toarray()
        X_test = X_test_text.toarray()
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def extract_linguistic_features_nb(text):
    """Extract linguistic features from text."""
    try:
        features = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_readability': textstat.dale_chall_readability_score(text),
            'gunning_fog': textstat.gunning_fog(text),
            'difficult_words': textstat.difficult_words(text),
            'lexicon_count': textstat.lexicon_count(text),
            'sentence_count': textstat.sentence_count(text)
        }
    except:
        features = {k: 0 for k in [
            'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index',
            'dale_chall_readability', 'gunning_fog', 'difficult_words',
            'lexicon_count', 'sentence_count'
        ]}
    return features

def load_and_clean_data_nb(train_csv_path="./data/text_train.csv",
                       val_csv_path="./data/text_validation.csv",
                       test_csv_path="./data/text_test.csv"):
    """Load and clean the dataset with default paths."""
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Data cleaning
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=['text', 'label'], inplace=True)
        df['label'] = df['label'].astype(int)
        df['text'] = df['text'].astype(str)

    return train_df, val_df, test_df

def preprocess_data_nb(use_linguistic_features=False,
                   train_csv_path="./data/text_train.csv",
                       val_csv_path="./data/text_validation.csv",
                       test_csv_path="./data/text_test.csv"):
    """Main preprocessing function with default paths."""
    train_df, val_df, test_df = load_and_clean_data_nb(
        train_csv_path, val_csv_path, test_csv_path
    )
    
    # Text vectorization
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7
    )
    X_train_text = vectorizer.fit_transform(train_df['text'])
    X_val_text = vectorizer.transform(val_df['text'])
    X_test_text = vectorizer.transform(test_df['text'])
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join("processed_data", "nb_vectorizer.joblib"))
    
    if use_linguistic_features:
        # Extract and scale linguistic features
        train_features = train_df['text'].apply(extract_linguistic_features_nb).apply(pd.Series)
        val_features = val_df['text'].apply(extract_linguistic_features_nb).apply(pd.Series)
        test_features = test_df['text'].apply(extract_linguistic_features_nb).apply(pd.Series)
        
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)
        
        # Combine features
        X_train = (X_train_text, train_features)
        X_val = (X_val_text, val_features)
        X_test = (X_test_text, test_features)
        
        # Save scaler
        joblib.dump(scaler, os.path.join("processed_data", "nb_scaler.joblib"))
    else:
        X_train = X_train_text
        X_val = X_val_text
        X_test = X_test_text
        scaler = None
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Initialize data
    data = initialize_data()

    # print(data.describe())
    print(data.info())

    # Check if text column contains duplicates
    if data['text'].duplicated().any():
        print("Duplicate text entries found. Removing duplicates...")
        data = data.drop_duplicates(subset=['text'])
    else:
        print("No duplicate text entries found.")

    print("Data after removing duplicates:")
    print(data.info())
    # print(data.describe())

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)
    print("Text data cleaned.")

    # Add linguistic features to the data
    features = data['text'].apply(extract_linguistic_features)
    features_df = pd.DataFrame(features.tolist())

    print("Linguistic features extracted.")
    print(features_df.head())
