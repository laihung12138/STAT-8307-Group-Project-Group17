import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import transformers
import peft
import accelerate
import argparse
import joblib
from transformers import AutoTokenizer
from models import (
    train_eval_distilbert_lora_no_features_model, 
    train_eval_distilbert_lora_with_features_model,
    train_and_evaluate_model_distilbert_f,
    train_and_evaluate_model_distilbert_feature,
    train_sgd, evaluate_model_sgd,
    train_naive_bayes, evaluate_model_nb
)
from model_training import main_lstm
from data_preprocessing import (
    data_preprocessing_original_file_dis_lora,
    split_raw_data_to_csv_dis_lora,
    data_preprocessing_freeze_last_layer_feature,
    data_preprocessing_freeze_last_layer_nofeature,
    preprocess_data_sgd, extract_linguistic_features_sgd,
    preprocess_data_nb, extract_linguistic_features_nb
)
from error_analysis import run_error_analysis
from sklearn.metrics import accuracy_score, f1_score

def main_dis_lora():
    split_raw_data_to_csv_dis_lora()
    data_preprocessing_original_file_dis_lora()
    print("Data preprocessing complete.")

    ##### Define Data Paths 
    PROCESSED_DATASET_NAME = 'processed_distilbert_data'
    INPUT_PATH = os.path.join('', PROCESSED_DATASET_NAME) 
    print(f"Looking for processed NumPy data files in: {INPUT_PATH}")

    # Define the expected .npy filenames 
    npy_files = [
        'train_ids.npy', 'train_mask.npy', 'train_features.npy', 'train_labels.npy',
        'val_ids.npy', 'val_mask.npy', 'val_features.npy', 'val_labels.npy',
        'test_ids.npy', 'test_mask.npy', 'test_features.npy', 'test_labels.npy',
    ]

    # Check if files exist 
    all_files_found = True
    for filename in npy_files:
        file_path = os.path.join(INPUT_PATH, filename)
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            all_files_found = False

    if not all_files_found:
        raise SystemExit(f"Please ensure all .npy files are present in the dataset '{PROCESSED_DATASET_NAME}' at {INPUT_PATH}")
    else:
        print("All required .npy files seem to be present.")


    ##### Configuration and Setup 
    # Configuration 
    MODEL_NAME = "distilbert-base-uncased"

    # NO_FEATURES_OUTPUT_DIR = "finetuned_model/distilbert_lora_no_features_finetuned_detector" ### original output dir
    NO_FEATURES_OUTPUT_DIR = "finetuned_model/distilbert_lora_no_features_finetuned_detector2" ### for try
    # NO_FEATURES_LOGGING_DIR = "logs/distilbert_lora_no_features_logs" ### original logging dir
    NO_FEATURES_LOGGING_DIR = "logs/distilbert_lora_no_features_logs2"  ### for try
    
    # WITH_FEATURES_OUTPUT_DIR = "finetuned_model/distilbert_lora_with_features_finetuned_detector"  ### original output dir
    WITH_FEATURES_OUTPUT_DIR = "finetuned_model/distilbert_lora_with_features_finetuned_detector2" ### for try
    # WITH_FEATURES_LOGGING_DIR = "logs/distilbert_lora_with_features_logs" ### original logging dir
    WITH_FEATURES_LOGGING_DIR = "logs/distilbert_lora_with_features_logs2" ### for try

    # LoRA Configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        "q_lin",
        "v_lin",
    ]

    # Training Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 3
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 512

    # Number of linguistic features
    NUM_LINGUISTIC_FEATURES = 8

    # Check for GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available. Using CPU.")

    # Create Output Directories
    for dir_path in [NO_FEATURES_OUTPUT_DIR, NO_FEATURES_LOGGING_DIR, 
                    WITH_FEATURES_OUTPUT_DIR, WITH_FEATURES_LOGGING_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    print("\nConfiguration setup complete.")


    ##### Load and Preprocess Data
    # Load NumPy arrays for each split 
    try:
        # Training Data
        train_ids = np.load(os.path.join(INPUT_PATH, 'train_ids.npy'))
        train_mask = np.load(os.path.join(INPUT_PATH, 'train_mask.npy'))
        train_features = np.load(os.path.join(INPUT_PATH, 'train_features.npy'))
        train_labels = np.load(os.path.join(INPUT_PATH, 'train_labels.npy'))

        # Validation Data
        val_ids = np.load(os.path.join(INPUT_PATH, 'val_ids.npy'))
        val_mask = np.load(os.path.join(INPUT_PATH, 'val_mask.npy'))
        val_features = np.load(os.path.join(INPUT_PATH, 'val_features.npy'))
        val_labels = np.load(os.path.join(INPUT_PATH, 'val_labels.npy'))

        # Test Data
        test_ids = np.load(os.path.join(INPUT_PATH, 'test_ids.npy'))
        test_mask = np.load(os.path.join(INPUT_PATH, 'test_mask.npy'))
        test_features = np.load(os.path.join(INPUT_PATH, 'test_features.npy'))
        test_labels = np.load(os.path.join(INPUT_PATH, 'test_labels.npy'))

        print("Successfully loaded all .npy arrays.")

    except Exception as e:
        raise SystemExit(f"Error loading .npy files: {e}")


    ##### Create datasets WITHOUT features
    print("\nCreating datasets WITHOUT features...")    
    # Create Hugging Face Dataset objects from the arrays (WITHOUT features)
    train_data_dict_no_features = {
        'input_ids': train_ids,
        'attention_mask': train_mask,
        'labels': train_labels     
    }
    val_data_dict_no_features = {
        'input_ids': val_ids,
        'attention_mask': val_mask,
        'labels': val_labels
    }
    test_data_dict_no_features = {
        'input_ids': test_ids,
        'attention_mask': test_mask,
        'labels': test_labels
    }

    try:
        train_dataset_no_features = Dataset.from_dict(train_data_dict_no_features)
        validation_dataset_no_features = Dataset.from_dict(val_data_dict_no_features)
        test_dataset_no_features = Dataset.from_dict(test_data_dict_no_features)
    except Exception as e:
        raise SystemExit(f"Error creating Dataset objects from dictionaries (no features): {e}")

    processed_datasets_no_features = DatasetDict({
        'train': train_dataset_no_features,
        'validation': validation_dataset_no_features,
        'test': test_dataset_no_features
    })

    # Set Format for PyTorch
    processed_datasets_no_features.set_format("torch")
    print("Created DatasetDict WITHOUT features.")


    ##### Create datasets WITH features
    print("\nCreating datasets WITH features...")
    # Create Hugging Face Dataset objects from the arrays (WITH features)
    train_data_dict_with_features = {
        'input_ids': train_ids,
        'attention_mask': train_mask,
        'features': train_features,
        'labels': train_labels     
    }
    val_data_dict_with_features = {
        'input_ids': val_ids,
        'attention_mask': val_mask,
        'features': val_features,
        'labels': val_labels
    }
    test_data_dict_with_features = {
        'input_ids': test_ids,
        'attention_mask': test_mask,
        'features': test_features,
        'labels': test_labels
    }

    try:
        train_dataset_with_features = Dataset.from_dict(train_data_dict_with_features)
        validation_dataset_with_features = Dataset.from_dict(val_data_dict_with_features)
        test_dataset_with_features = Dataset.from_dict(test_data_dict_with_features)
    except Exception as e:
        raise SystemExit(f"Error creating Dataset objects from dictionaries (with features): {e}")

    processed_datasets_with_features = DatasetDict({
        'train': train_dataset_with_features,
        'validation': validation_dataset_with_features,
        'test': test_dataset_with_features
    })

    # Set Format for PyTorch
    processed_datasets_with_features.set_format("torch")
    print("Created DatasetDict WITH features.")

    # Print dataset stats
    for ds_name, ds in [("No Features", processed_datasets_no_features), 
                        ("With Features", processed_datasets_with_features)]:
        print(f"\n{ds_name} dataset structure:")
        print(ds)
        print(f"Training set size: {len(ds['train'])}")
        print(f"Validation set size: {len(ds['validation'])}")
        print(f"Test set size: {len(ds['test'])}")
        print(f"Columns in training set: {ds['train'].column_names}")


    ##### Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    ##### Choose which model to run
    model_choice = input("Which distilbert with lora model do you want to run? Enter 1 for model without features, 2 for model with features, or 3 for both: ")
    
    if model_choice == "1" or model_choice == "3":
        ##### Run Model Training and Evaluation (WITHOUT FEATURES)
        print("\n\n========== TRAINING MODEL WITHOUT FEATURES ==========")
        trainer_no_features, test_results_no_features = train_eval_distilbert_lora_no_features_model(
            tokenizer=tokenizer,
            train_dataset=processed_datasets_no_features["train"],
            validation_dataset=processed_datasets_no_features["validation"],
            test_dataset=processed_datasets_no_features["test"],
            model_name=MODEL_NAME,
            output_dir=NO_FEATURES_OUTPUT_DIR,
            logging_dir=NO_FEATURES_LOGGING_DIR,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY
        )
        print("\nModel without features evaluation complete!")

    if model_choice == "2" or model_choice == "3":
        ##### Run Model Training and Evaluation (WITH FEATURES)
        print("\n\n========== TRAINING MODEL WITH FEATURES ==========")
        trainer_with_features, test_results_with_features = train_eval_distilbert_lora_with_features_model(
            tokenizer=tokenizer,
            train_dataset=processed_datasets_with_features["train"],
            validation_dataset=processed_datasets_with_features["validation"],
            test_dataset=processed_datasets_with_features["test"],
            model_name=MODEL_NAME,
            output_dir=WITH_FEATURES_OUTPUT_DIR,
            logging_dir=WITH_FEATURES_LOGGING_DIR,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            num_linguistic_features=NUM_LINGUISTIC_FEATURES
        )
        print("\nModel with features evaluation complete!")

    print("\nAll evaluations complete!")

def main_dis_last_layer():
    # 读取数据
    data = pd.read_csv("./data/data_merged_original_yuke.csv")
    print("Data loaded successfully.")

    # Choose which model to run
    model_choice = input("Which distilbert with lora model do you want to run? Enter 1 for model without features, 2 for model with features: ")
    if model_choice == "1":
    
        #没有feature的数据处理
        train_dataset, val_dataset, test_dataset, y_train, y_val, y_test, datas_train, datas_val, datas_test = data_preprocessing_freeze_last_layer_nofeature(data)
        print("Data preprocessing completed successfully.")
        
        accuracy, f1, model = train_and_evaluate_model_distilbert_f(train_dataset, val_dataset, y_val)
        #测试测试集
        predictions = model.predict(test_dataset)
        pred_labels = predictions.predictions.argmax(axis=1)
        accuracy = accuracy_score(y_test, pred_labels)
        f1_s = f1_score(y_test, pred_labels, average='weighted')

        print(f"准确率: {accuracy}")
        print(f"F1分数: {f1_s}")
        
        #准备验证集数据进行error分析
        predictions = model.predict(val_dataset)
        pred_labels = predictions.predictions.argmax(axis=1)

        logits = predictions.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predicted_labels = np.argmax(logits, axis=1)
        predicted_probabilities = probabilities[np.arange(len(predicted_labels)), predicted_labels] # Probability of the predicted class

        val_df = pd.DataFrame()
        val_df['text'] = datas_val
        val_df['label'] = y_val
        val_df['prediction'] = pred_labels
        val_df['pred_probability'] = predicted_probabilities
        val_df['is_correct'] = (val_df['label'] == val_df['prediction'])
        
        #确保有result_type
        # Identify TP, TN, FP, FN
        val_df['result_type'] = 'NA'
        val_df.loc[(val_df['is_correct'] == True) & (val_df['label'] == 0), 'result_type'] = 'TN' 
        val_df.loc[(val_df['is_correct'] == True) & (val_df['label'] == 1), 'result_type'] = 'TP'
        val_df.loc[(val_df['is_correct'] == False) & (val_df['label'] == 0), 'result_type'] = 'FP'
        val_df.loc[(val_df['is_correct'] == False) & (val_df['label'] == 1), 'result_type'] = 'FN'
    
    elif model_choice == "2":       
        #有feature的数据处理
        train_dataset, val_dataset, test_dataset, y_train, y_val, y_test, datas_train, datas_val, datas_test = data_preprocessing_freeze_last_layer_feature(data)
        print("Data preprocessing completed successfully.")
        
        #有feature的模型
        accuracy, f1, model = train_and_evaluate_model_distilbert_feature(train_dataset, val_dataset, y_val)
        
        #测试测试集
        predictions = model.predict(test_dataset)
        pred_labels = predictions.predictions.argmax(axis=1)
        accuracy = accuracy_score(y_test, pred_labels)
        f1_s = f1_score(y_test, pred_labels, average='weighted')

        print(f"准确率: {accuracy}")
        print(f"F1分数: {f1_s}")
        
        #准备验证集数据进行error分析
        predictions = model.predict(val_dataset)
        pred_labels = predictions.predictions.argmax(axis=1)

        logits = predictions.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predicted_labels = np.argmax(logits, axis=1)
        predicted_probabilities = probabilities[np.arange(len(predicted_labels)), predicted_labels] # Probability of the predicted class

        val_df = pd.DataFrame()
        val_df['text'] = datas_val
        val_df['label'] = y_val
        val_df['prediction'] = pred_labels
        val_df['pred_probability'] = predicted_probabilities
        val_df['is_correct'] = (val_df['label'] == val_df['prediction'])

        
        #确保有result_type
        # Identify TP, TN, FP, FN
        val_df['result_type'] = 'NA'
        val_df.loc[(val_df['is_correct'] == True) & (val_df['label'] == 0), 'result_type'] = 'TN' 
        val_df.loc[(val_df['is_correct'] == True) & (val_df['label'] == 1), 'result_type'] = 'TP'
        val_df.loc[(val_df['is_correct'] == False) & (val_df['label'] == 0), 'result_type'] = 'FP'
        val_df.loc[(val_df['is_correct'] == False) & (val_df['label'] == 1), 'result_type'] = 'FN'

def main_sgd():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train and evaluate SGD model for AI text detection")
    parser.add_argument('--train_path', default="./data/text_train.csv", help="Path to training data")
    parser.add_argument('--val_path', default="./data/text_validation.csv", help="Path to validation data")
    parser.add_argument('--test_path', default="./data/text_test.csv", help="Path to test data")
    parser.add_argument('--output_dir', default="results", help="Output directory for results")
    args = parser.parse_args()

    # Choose which model to run
    model_choice = input("Which distilbert with lora model do you want to run? Enter 1 for model without features, 2 for model with features: ")
    if model_choice == "1":
        features_addition = False
    elif model_choice == "2":
        features_addition = True

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    
    # Preprocess data
    print("Preprocessing data...")
    split_raw_data_to_csv_dis_lora()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data_sgd(
        use_linguistic_features=features_addition,
        train_csv_path=args.train_path,
        val_csv_path=args.val_path,
        test_csv_path=args.test_path
    )
    
    # Train model
    model = train_sgd(X_train, y_train, features_addition)
    
    # Evaluate on all sets
    print("\nEvaluating model...")
    print("\nTraining Set Performance:")
    train_results = evaluate_model_sgd(model, X_train, y_train, "Training", features_addition)
    
    print("\nValidation Set Performance:")
    val_results = evaluate_model_sgd(model, X_val, y_val, "Validation", features_addition)
    
    print("\nTest Set Performance:")
    test_results = evaluate_model_sgd(model, X_test, y_test, "Test", features_addition)
    
    # Save all results to CSV
    results_df = pd.DataFrame({
        'Dataset': ['Training', 'Validation', 'Test'],
        'Accuracy': [train_results['accuracy'], val_results['accuracy'], test_results['accuracy']],
        'F1_Score': [train_results['f1'], val_results['f1'], test_results['f1']],
        'ROC_AUC': [train_results['roc_auc'], val_results['roc_auc'], test_results['roc_auc']]
    })
    
    output_dir = os.path.join(args.output_dir, "sgd_with_features" if features_addition else "sgd_no_features")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    
    # Prepare validation data for error analysis
    val_df = pd.read_csv(args.val_path)
    val_df = val_df.dropna(subset=['text', 'label'])
    val_df['label'] = val_df['label'].astype(int)
    
    # Load vectorizer
    vectorizer = joblib.load("processed_data/sgd_vectorizer.joblib")
    
    # Generate predictions
    if features_addition:
        # Load scaler if using linguistic features
        scaler = joblib.load("processed_data/sgd_scaler.joblib")
        text_features = vectorizer.transform(val_df['text']).toarray()
        ling_features = scaler.transform(
            val_df['text'].apply(extract_linguistic_features_sgd).apply(pd.Series)
        )
        
        combined_features = np.hstack([text_features, ling_features])
        val_df['prediction'] = model.predict(combined_features)
        val_df['pred_probability'] = model.predict_proba(combined_features)[:, 1]
    else:
        text_features = vectorizer.transform(val_df['text']).toarray()
        val_df['prediction'] = model.predict(text_features)
        val_df['pred_probability'] = model.predict_proba(text_features)[:, 1]
    
    # Add analysis columns
    val_df['is_correct'] = val_df['label'] == val_df['prediction']
    val_df['result_type'] = val_df.apply(
        lambda x: 'TP' if x['label']==1 and x['is_correct'] else 
                 ('TN' if x['label']==0 and x['is_correct'] else
                 ('FP' if x['label']==0 else 'FN')), axis=1)
    
    # Run error analysis
    print("\nRunning error analysis...")
    run_error_analysis(val_df, output_dir, features_addition)
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {output_dir}")


def main_nb():
    parser = argparse.ArgumentParser(description="Train and evaluate Naive Bayes model for AI text detection")
    parser.add_argument('--train_path', default="./data/text_train.csv", help="Path to training data")
    parser.add_argument('--val_path', default="./data/text_validation.csv", help="Path to validation data")
    parser.add_argument('--test_path', default="./data/text_test.csv", help="Path to test data")
    parser.add_argument('--output_dir', default="results", help="Output directory for results")
    args = parser.parse_args()

    # Choose which model to run
    model_choice = input("Which distilbert with lora model do you want to run? Enter 1 for model without features, 2 for model with features: ")
    if model_choice == "1":
        features_addition = False
    elif model_choice == "2":
        features_addition = True

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    
    # Preprocess data
    print("Preprocessing data...")
    split_raw_data_to_csv_dis_lora()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data_nb(
        use_linguistic_features=features_addition,
        train_csv_path=args.train_path,
        val_csv_path=args.val_path,
        test_csv_path=args.test_path
    )
    
    # Train model
    model = train_naive_bayes(X_train, y_train, features_addition)
    
    # Evaluate on all sets
    print("\nEvaluating model...")
    print("\nTraining Set Performance:")
    train_results = evaluate_model_nb(model, X_train, y_train, "Training", features_addition)
    
    print("\nValidation Set Performance:")
    val_results = evaluate_model_nb(model, X_val, y_val, "Validation", features_addition)
    
    print("\nTest Set Performance:")
    test_results = evaluate_model_nb(model, X_test, y_test, "Test", features_addition)
    
    # Save all results to CSV
    results_df = pd.DataFrame({
        'Dataset': ['Training', 'Validation', 'Test'],
        'Accuracy': [train_results['accuracy'], val_results['accuracy'], test_results['accuracy']],
        'F1_Score': [train_results['f1'], val_results['f1'], test_results['f1']],
        'ROC_AUC': [train_results['roc_auc'], val_results['roc_auc'], test_results['roc_auc']]
    })
    
    output_dir = os.path.join(args.output_dir, "nb_with_features" if features_addition else "nb_no_features")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    
    # Prepare validation data for error analysis
    val_df = pd.read_csv(args.val_path)
    val_df = val_df.dropna(subset=['text', 'label'])
    val_df['label'] = val_df['label'].astype(int)
    
    # Load necessary objects
    vectorizer = joblib.load("processed_data/nb_vectorizer.joblib")
    if features_addition:
        scaler = joblib.load("processed_data/nb_scaler.joblib")
    
    # Generate predictions
    if features_addition:
        text_features = vectorizer.transform(val_df['text'])
        ling_features = scaler.transform(
            val_df['text'].apply(extract_linguistic_features_nb).apply(pd.Series)
        )
        val_df['prediction'] = model.predict((text_features, ling_features))
        val_df['pred_probability'] = model.predict_proba((text_features, ling_features))[:, 1]
    else:
        val_df['prediction'] = model.predict(vectorizer.transform(val_df['text']))
        val_df['pred_probability'] = model.predict_proba(vectorizer.transform(val_df['text']))[:, 1]
    
    # Add analysis columns
    val_df['is_correct'] = val_df['label'] == val_df['prediction']
    val_df['result_type'] = val_df.apply(
        lambda x: 'TP' if x['label']==1 and x['is_correct'] else 
                 ('TN' if x['label']==0 and x['is_correct'] else
                 ('FP' if x['label']==0 else 'FN')), axis=1)
    
    # Run error analysis
    print("\nRunning error analysis...")
    run_error_analysis(val_df, output_dir, features_addition)
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Ask user for model choice
    model_choice = input("Which model do you want to run? Enter 1 for distilbert with lora, 2 for distilbert with last layer, 3 for sgd, 4 for nb, or 5 for lstm: ")

    if model_choice == "1":
        main_dis_lora()
    elif model_choice == "2":
        main_dis_last_layer()
    elif model_choice == "3":
        main_sgd()
    elif model_choice == "4":
        main_nb()
    elif model_choice == "5":
        main_lstm()
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")