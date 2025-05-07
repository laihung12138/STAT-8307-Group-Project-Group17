import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, DistilBertModel, AutoModelForSequenceClassification
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
import torch
import os
import evaluate
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertPreTrainedModel, DistilBertModel, AutoConfig
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,DistilBertPreTrainedModel
from transformers import DistilBertConfig
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from transformers.modeling_outputs import SequenceClassifierOutput 
import time
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import  AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput 

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, BatchNormalization

def build_lstm_model(max_words, max_length, features_addition, lstm_units=128, lstm_layers=5, dropout_rate=0.4):
    # Input layer for word-level features
    word_input = Input(shape=(max_length,), name="word_level_input")
    x_word = Embedding(input_dim=max_words, output_dim=128)(word_input)

    # Add multiple LSTM layers
    for i in range(lstm_layers):
        x_word = LSTM(lstm_units, return_sequences=(i < lstm_layers - 1))(x_word)  # Return sequences for all but the last layer
        x_word = BatchNormalization()(x_word)  # Normalization for better training stability
        x_word = Dropout(dropout_rate)(x_word)  # Dropout to prevent overfitting

    if features_addition:
        # Input for additional linguistic features
        linguistic_input = Input(shape=(8,), name="linguistic_input")
        x_linguistic = Dense(256, activation="relu")(linguistic_input)
        x_linguistic = Dropout(dropout_rate)(x_linguistic)  # Dropout for regularization

        # Concatenate both branches
        x = Concatenate()([x_word, x_linguistic])
    else:
        x = x_word

    # Fully connected layers
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(1, activation="sigmoid")(x)

    # Create model
    if features_addition:
        model = Model(inputs=[word_input, linguistic_input], outputs=output)
    else:
        model = Model(inputs=word_input, outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def compute_metrics(p: EvalPrediction):
    # Extract logits and labels
    logits = p.predictions
    labels = p.label_ids

    # Calculate Predicted Classes
    preds = np.argmax(logits, axis=-1)

    # Calculate Probabilities for AUC 
    try:
        probs = softmax(logits, axis=-1)[:, 1]
    except IndexError:
        print("Warning: Logits shape might not be [batch_size, 2]. Adjusting AUC probability calculation.")
        probs = np.full(labels.shape, np.nan)

    # Calculate Metrics 
    # Accuracy
    acc = accuracy_score(labels, preds)

    # F1 Score
    f1 = f1_score(labels, preds, average='binary')

    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError as e:
         print(f"Warning: Could not calculate AUC. Error: {e}. Returning NaN for AUC.")
         auc = np.nan 

    # Confusion Matrix 
    cm = confusion_matrix(labels, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"Warning: Confusion matrix shape unexpected ({cm.shape}). Setting TN/FP/FN/TP to NaN.")
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }

def train_eval_distilbert_lora_no_features_model(
        tokenizer,
        train_dataset,
        validation_dataset,
        test_dataset,
        model_name="distilbert-base-uncased",
        output_dir="finetuned_model/distilbert_lora_no_features_finetuned_detector",
        logging_dir="logs/distilbert_lora_no_features_logs",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q_lin", "v_lin"],
        learning_rate=1e-4,
        batch_size=16,
        epochs=3,
        weight_decay=0.01
    ):
    
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification
    )

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )

    print("Applying LoRA to the model...")
    lora_model = get_peft_model(model, lora_config)

    print("\nTrainable parameters after applying LoRA:")
    lora_model.print_trainable_parameters()

    # Training Arguments
    print("\nDefining Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        # evaluation_strategy="epoch",
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 to select best model
        greater_is_better=True,
        push_to_hub=False,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        label_names=["labels"]
    )

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # Instantiate Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Setup complete. Ready for training.")

    # Train the model
    print("Starting LoRA model training...")
    start_time = time.time()

    train_result = trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Log and Save Final Metrics 
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save the LoRA adapter weights explicitly
    final_adapter_path = os.path.join(output_dir, "final_lora_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Final LoRA adapter weights saved to: {final_adapter_path}")

    # Save tokenizer
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to: {output_dir}")

    print("\nTraining process complete. Best model saved based on validation F1 score.")
    
    # Evaluate on test set
    print("Starting evaluation on the test set using the best model checkpoint...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n--- Test Set Evaluation Results ---")
    for key, value in test_results.items():
        metric_name = key.replace("eval_", "")
        if metric_name in ['tn', 'fp', 'fn', 'tp']:
            print(f"{metric_name}: {int(value):,}")
        elif isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
            
    # Save Test Results 
    import json
    test_results_path = os.path.join(output_dir, "test_results.json")
    results_to_save = {f"test_{k.replace('eval_','')}": v for k, v in test_results.items()}

    with open(test_results_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"\nTest results saved to {test_results_path}")
    
    return trainer, test_results

# Custom model class for DistilBERT with linguistic features
class DistilBertWithFeatures(DistilBertPreTrainedModel):
    """
    DistilBERT model for sequence classification that incorporates additional
    numerical features alongside the text embeddings.
    """
    def __init__(self, config, num_linguistic_features):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_linguistic_features = num_linguistic_features
        self.config = config

        # Base DistilBERT model
        self.distilbert = DistilBertModel(config)

        ### Custom Classification Head 
        # Calculate the combined size: DistilBERT hidden size + number of linguistic features
        combined_feature_size = config.dim + self.num_linguistic_features

        # Dropout layer 
        self.pre_classifier_dropout = nn.Dropout(config.seq_classif_dropout)

        # Final classifier layer mapping the combined features to logits
        self.classifier = nn.Linear(combined_feature_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        features=None,  # Accept the linguistic features 
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass that combines DistilBERT output with linguistic features.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get outputs from the base DistilBERT model
        distilbert_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the [CLS] token embedding
        # The last hidden state is the first element in the output tuple/dict
        hidden_state = distilbert_outputs[0]  # (batch_size, seq_len, dim)
        cls_embedding = hidden_state[:, 0]    # (batch_size, dim) - embedding of the first token ([CLS])

        ### Feature Combination 
        if features is None:
             raise ValueError("Linguistic 'features' must be provided to DistilBertWithFeatures forward pass.")
        features = features.type_as(cls_embedding) 

        # Concatenate the [CLS] embedding with the linguistic features
        combined_features = torch.cat((cls_embedding, features), dim=1) 

        # Apply dropout
        combined_features_dropped = self.pre_classifier_dropout(combined_features)

        # Pass combined features through the classifier to get logits
        logits = self.classifier(combined_features_dropped) 

        ### Loss Calculation 
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        ### Prepare Output
        if not return_dict:
            output = (logits,) + distilbert_outputs[1:] 
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_outputs.hidden_states, 
            attentions=distilbert_outputs.attentions,     
        )
    
# Train and Evaluate DistilBert with lora model (with features added) Function
def train_eval_distilbert_lora_with_features_model(
        tokenizer,
        train_dataset,
        validation_dataset,
        test_dataset,
        model_name="distilbert-base-uncased",
        output_dir="finetuned_model/distilbert_lora_with_features_finetuned_detector",
        logging_dir="logs/distilbert_lora_with_features_logs",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q_lin", "v_lin"],
        learning_rate=1e-4,
        batch_size=16,
        epochs=3,
        weight_decay=0.01,
        num_linguistic_features=8
    ):
    
    # Load Model Configuration 
    print(f"Loading configuration for {model_name}...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification
    )

    # Instantiate CUSTOM Model 
    print(f"Instantiating custom model 'DistilBertWithFeatures' with {num_linguistic_features} linguistic features...")
    model = DistilBertWithFeatures.from_pretrained(
        model_name,
        config=config,  # Pass the loaded config
        num_linguistic_features=num_linguistic_features  # Pass the number of features
    )
    print("Custom model loaded.")

    # Configure LoRA 
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )

    print("Applying LoRA to the custom model...")
    lora_model = get_peft_model(model, lora_config)

    print("\nTrainable parameters after applying LoRA to custom model:")
    lora_model.print_trainable_parameters()

    # Training Arguments
    print("\nDefining Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        # evaluation_strategy="epoch",
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 to select best model
        greater_is_better=True,
        push_to_hub=False,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        label_names=["labels"]
    )

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # Instantiate Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Setup complete. Ready for training.")

    # Train the model
    print("Starting LoRA model training...")
    start_time = time.time()

    train_result = trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Log and Save Final Metrics 
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save the LoRA adapter weights explicitly
    final_adapter_path = os.path.join(output_dir, "final_lora_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Final LoRA adapter weights saved to: {final_adapter_path}")

    # Save tokenizer
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to: {output_dir}")

    print("\nTraining process complete. Best model saved based on validation F1 score.")
    
    # Evaluate on test set
    print("Starting evaluation on the test set using the best model checkpoint...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n--- Test Set Evaluation Results ---")
    for key, value in test_results.items():
        metric_name = key.replace("eval_", "")
        if metric_name in ['tn', 'fp', 'fn', 'tp']:
            print(f"{metric_name}: {int(value):,}")
        elif isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
            
    # Save Test Results 
    test_results_path = os.path.join(output_dir, "test_results.json")
    results_to_save = {f"test_{k.replace('eval_','')}": v for k, v in test_results.items()}

    with open(test_results_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"\nTest results saved to {test_results_path}")
    
    return trainer, test_results

# =======================================================================================================

class DistilBertWithFeatures_last_layer(DistilBertPreTrainedModel):
    """
    DistilBERT model for sequence classification that incorporates additional
    numerical features alongside the text embeddings.
    """
    def __init__(self, config, num_linguistic_features):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_linguistic_features = num_linguistic_features
        self.config = config

        # Base DistilBERT model
        self.distilbert = DistilBertModel(config)

        # --- Custom Classification Head ---
        # Calculate the combined size: DistilBERT hidden size + number of linguistic features
        combined_feature_size = config.dim + self.num_linguistic_features

        # Dropout layer 
        self.pre_classifier_dropout = nn.Dropout(config.seq_classif_dropout)

        # Final classifier layer mapping the combined features to logits
        self.classifier = nn.Linear(combined_feature_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        features=None,  # Accept the linguistic features 
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass that combines DistilBERT output with linguistic features.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get outputs from the base DistilBERT model
        distilbert_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the [CLS] token embedding
        # The last hidden state is the first element in the output tuple/dict
        hidden_state = distilbert_outputs[0]  # (batch_size, seq_len, dim)
        cls_embedding = hidden_state[:, 0]    # (batch_size, dim) - embedding of the first token ([CLS])

        # --- Feature Combination ---
        # Ensure linguistic features have the correct type (float)
        if features is None:
             raise ValueError("Linguistic 'features' must be provided to DistilBertWithFeatures_last_layer forward pass.")
        features = features.type_as(cls_embedding) 

        # Concatenate the [CLS] embedding with the linguistic features
        combined_features = torch.cat((cls_embedding, features), dim=1) # (batch_size, dim + num_linguistic_features)

        # Apply dropout
        combined_features_dropped = self.pre_classifier_dropout(combined_features)

        # Pass combined features through the classifier to get logits
        logits = self.classifier(combined_features_dropped) # (batch_size, num_labels)

        # --- Loss Calculation ---
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # --- Prepare Output ---
        if not return_dict:
            output = (logits,) + distilbert_outputs[1:] 
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_outputs.hidden_states, 
            attentions=distilbert_outputs.attentions,     
        )

    print("Custom model class 'DistilBertWithFeatures_last_layer' defined.")

def train_and_evaluate_model_distilbert_f(train_dataset, val_dataset, y_val):
    MODEL_NAME = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    for name, param in model.named_parameters():
        param.requires_grad = False

    last_layer = model.distilbert.transformer.layer[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    wandb.login(key="519e2e9a6ed1abb94f1d1c5ed4f270675ab1bd21")
    
    training_args = TrainingArguments(
        output_dir='./results',          
        run_name="my_experiment",
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,   
        warmup_steps=500,               
        weight_decay=0.01,               
        logging_dir='./logs',            
        eval_strategy="epoch",     
        save_strategy="epoch",  
    )

    trainer = Trainer(
        model=model,                    
        args=training_args,             
        train_dataset=train_dataset,    
        eval_dataset=val_dataset        
    )

    trainer.train()

    predictions = trainer.predict(val_dataset)
    pred_labels = predictions.predictions.argmax(axis=1)
    accuracy = accuracy_score(y_val, pred_labels)
    f1 = f1_score(y_val, pred_labels, average='weighted')

    print(f"准确率: {accuracy}")
    print(f"F1分数: {f1}")

    return accuracy, f1,trainer

# 使用示例
def train_and_evaluate_model_distilbert_feature(train_dataset, val_dataset, y_val):
    MODEL_NAME = "distilbert-base-uncased"
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2, # Binary classification
    )
    
    # --- Instantiate CUSTOM Model ---
    NUM_LINGUISTIC_FEATURES = 8
    print(f"Instantiating custom model 'DistilBertWithFeatures' with {NUM_LINGUISTIC_FEATURES} linguistic features...")
    model = DistilBertWithFeatures.from_pretrained(
        MODEL_NAME,
        config=config, # Pass the loaded config
        num_linguistic_features=NUM_LINGUISTIC_FEATURES # Pass the number of features
    )
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    print("Custom model loaded.")

    for name, param in model.named_parameters():
        param.requires_grad = False

    last_layer = model.distilbert.transformer.layer[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    wandb.login(key="519e2e9a6ed1abb94f1d1c5ed4f270675ab1bd21")
    
    training_args = TrainingArguments(
        output_dir='./results',          
        run_name="my_experiment",
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,   
        warmup_steps=500,               
        weight_decay=0.01,               
        logging_dir='./logs',            
        eval_strategy="epoch",     
        save_strategy="epoch",  
    )

    trainer = Trainer(
        model=model,                    
        args=training_args,             
        train_dataset=train_dataset,    
        eval_dataset=val_dataset        
    )

    trainer.train()

    predictions = trainer.predict(val_dataset)
    pred_labels = predictions.predictions.argmax(axis=1)
    accuracy = accuracy_score(y_val, pred_labels)
    f1 = f1_score(y_val, pred_labels, average='weighted')

    print(f"准确率: {accuracy}")
    print(f"F1分数: {f1}")

    return accuracy, f1,trainer

def train_sgd(X_train, y_train, use_linguistic_features=False):
    """Train optimized SGD model with fixed learning rate"""
    print("\nTraining SGD model...")
    start_time = time.time()
    
    # Configure SGD with robust settings
    model = SGDClassifier(
        loss='log_loss',          # Logistic regression
        penalty='elasticnet',     # L1 + L2 regularization
        alpha=1e-4,               # Regularization strength
        l1_ratio=0.15,            # Balance L1/L2
        max_iter=1000,            # Increased iterations
        random_state=42,
        learning_rate='optimal',  # Replaced 'adaptive' with 'optimal'
        eta0=0.01,                # Explicit initial learning rate
        early_stopping=False,     # Simpler configuration
        class_weight='balanced',  # Handle imbalanced data
        tol=1e-4                  # Optimization tolerance
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_type = "with_features" if use_linguistic_features else "no_features"
    os.makedirs(f"models/sgd_{model_type}", exist_ok=True)
    joblib.dump(model, f"models/sgd_{model_type}/model.joblib")
    
    return model

def evaluate_model_sgd(model, X, y, set_name="Dataset", use_features=False):
    """Evaluation with metrics and confusion matrix"""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    
    # Core metrics
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    roc_auc = roc_auc_score(y, proba)
    cm = confusion_matrix(y, pred)
    
    # Print results
    print(f"\n{set_name} Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], 
                yticklabels=['Human', 'AI'])
    plt.title(f'{set_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


class NaiveBayesWithFeatures:
    """Custom Naive Bayes that combines text and linguistic features"""
    def __init__(self):
        self.text_nb = MultinomialNB()
        self.features_nb = GaussianNB()
        self.feature_weights = [0.7, 0.3]  # Weighted probability combination
        
    def fit(self, X, y):
        text_features, ling_features = X
        self.text_nb.fit(text_features, y)
        self.features_nb.fit(ling_features, y)
        return self
        
    def predict_proba(self, X):
        text_features, ling_features = X
        text_proba = self.text_nb.predict_proba(text_features)
        ling_proba = self.features_nb.predict_proba(ling_features)
        return (self.feature_weights[0]*text_proba + 
                self.feature_weights[1]*ling_proba)
    
    def predict(self, X):
        """Required predict method that uses predict_proba"""
        return np.argmax(self.predict_proba(X), axis=1)

def train_naive_bayes(X_train, y_train, use_linguistic_features=False):
    """Train Naive Bayes model with option for linguistic features"""
    print("\nTraining Naive Bayes model...")
    start_time = time.time()
    
    if use_linguistic_features:
        model = NaiveBayesWithFeatures()
        model.fit(X_train, y_train)
    else:
        model = MultinomialNB()
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_type = "with_features" if use_linguistic_features else "no_features"
    os.makedirs(f"models/nb_{model_type}", exist_ok=True)
    joblib.dump(model, f"models/nb_{model_type}/model.joblib")
    
    return model

def evaluate_model_nb(model, X, y, set_name="Dataset", use_features=False):
    """Evaluate model performance"""
    if use_features and isinstance(model, NaiveBayesWithFeatures):
        # Handle custom NB with features
        if isinstance(X, tuple):  # Combined features case
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1]
        else:  # Text-only fallback
            pred = model.text_nb.predict(X)
            proba = model.text_nb.predict_proba(X)[:, 1]
    else:  # Standard NB case
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    roc_auc = roc_auc_score(y, proba)
    cm = confusion_matrix(y, pred)
    
    # Print results
    print(f"\n{set_name} Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], 
                yticklabels=['Human', 'AI'])
    plt.title(f'{set_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
