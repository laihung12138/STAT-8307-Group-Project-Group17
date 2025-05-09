
# STAT-8307 Group Project: Group 17

This repository contains the code for the STAT-8307 group project. The project involves training and evaluating various machine learning models for text classification tasks. Each script in the repository has a distinct purpose, ranging from data preprocessing to exploratory data analysis (EDA) and model training.

---

## Repository Structure

The repository is organized as follows:

- **`main.py`**: The primary script for running all models. It provides an interactive interface to choose the desired model and configuration.
- **`eda.py`**: A dedicated script for exploratory data analysis (EDA) to understand the dataset.
- **`data_preprocessing.py`**: Functions for cleaning and preparing the raw data for training.
- **`distilbert_lora_inference_utils.py`**: Utilities for inference using DistilBERT models trained with LoRA (Low-Rank Adaptation).
- **`error_analysis.py`**: Tools for analyzing model errors and visualizing the results.
- **`model_training.py`**: Contains training functions for LSTM models.
- **`models.py`**: Defines the architecture of machine learning models, including DistilBERT, SGD, Naive Bayes, and LSTM.
- **`split_raw_data_to_csv.py`**: Splits the raw dataset into training, validation, and test subsets.

---

## How to Use

### 1. Retrieve datasets from Hugging Face

To download datasets from hugging face, run the `Retrieve_data_from_hugging_face.py` script:

```bash
python Retrieve_data_from_hugging_face.py
```

### 2. Exploratory Data Analysis (EDA)

To perform EDA on the dataset, run the `eda.py` script:

```bash
python eda.py
```

The script will output visualizations and summary statistics, helping to understand the data distribution and characteristics.

---

### 3. Running Models

To train and evaluate models, execute the `main.py` script:

```bash
python main.py

```
Upon running, the script will prompt you to select a model. Choose from the following options:

```
Enter 1 for distilbert with lora, 
2 for distilbert with last layer, 
3 for sgd, 
4 for nb, or 
5 for lstm
```

After selecting a model, the script will further ask if you want to include linguistic features in the training process. For example:

```
Which distilbert with lora model do you want to run? 
Enter 1 for model without features, 
2 for model with features
```

Once the choices are made, the script will preprocess the data, train the selected model, and evaluate its performance. Results will be displayed in the terminal and saved in the output directories.

---

## Requirements

Ensure the following dependencies are installed:

- **Python**: Version 3.9 or higher

---

## Models Available

The following models are supported in this project:

1. **DistilBERT with LoRA**: Fine-tunes a DistilBERT model using LoRA for efficient parameter updates.
2. **DistilBERT with Last Layer**: Fine-tunes only the last layer of the DistilBERT model.
3. **SGD**: A linear model trained using Stochastic Gradient Descent.
4. **Naive Bayes (NB)**: A simple and efficient probabilistic classifier.
5. **LSTM**: A Long Short-Term Memory network for sequence modeling.

---
