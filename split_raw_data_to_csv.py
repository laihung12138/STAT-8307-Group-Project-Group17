import numpy as np
import pandas as pd
import json
import re
import textstat
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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


if __name__ == "__main__":
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