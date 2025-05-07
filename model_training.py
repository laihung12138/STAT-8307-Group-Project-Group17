import pandas as pd
from data_preprocessing import preprocess_data
from models import build_lstm_model
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import Callback


class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        predictions = self.model.predict(val_data)
        predictions = (predictions > 0.5).astype(int)  # Assuming binary classification

        # Calculate metrics
        f1 = f1_score(val_labels, predictions)
        auc = roc_auc_score(val_labels, predictions)
        tn, fp, fn, tp = confusion_matrix(val_labels, predictions).ravel()

        # Log metrics to console
        print(f"\nEpoch {epoch + 1} Metrics:")
        print(f" - F1 Score: {f1:.4f}")
        print(f" - AUC: {auc:.4f}")
        print(f" - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

        # Optionally add metrics to logs (for TensorBoard or other callbacks)
        logs.update({"f1": f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp})


def main_lstm():
    print("Loading data...")

    # Choose which model to run
    model_choice = input("Which distilbert with lora model do you want to run? Enter 1 for model without features, 2 for model with features: ")
    if model_choice == "1":
        features_addition = False
    elif model_choice == "2":
        features_addition = True

    print("Preprocessing data...")
    max_words = 10000
    max_length = 100

    if features_addition:
        (X_train_word, X_test_word, X_val_word,
         X_train_linguistic, X_test_linguistic, X_val_linguistic,
         y_train, y_test, y_val,
         tokenizer) = preprocess_data(max_words, max_length)

        print("Building model...")
        model = build_lstm_model(max_words, max_length, features_addition)

        print("Training model...")
        metrics_callback = MetricsCallback(validation_data=(
            {"word_level_input": X_val_word, "linguistic_input": X_val_linguistic}, y_val
        ))

        model.fit(
            {"word_level_input": X_train_word, "linguistic_input": X_train_linguistic},
            y_train,
            epochs=10,
            batch_size=128,
            validation_data=({"word_level_input": X_val_word, "linguistic_input": X_val_linguistic}, y_val),
            callbacks=[metrics_callback],
            verbose=1
        )

        print("Evaluating model...")
        loss, accuracy = model.evaluate(
            {"word_level_input": X_test_word, "linguistic_input": X_test_linguistic},
            y_test
        )
    else:
        (X_train_word, X_test_word, X_val_word,
         y_train, y_test, y_val,
         tokenizer) = preprocess_data(max_words, max_length, features_addition)

        print("Building model...")
        model = build_lstm_model(max_words, max_length, features_addition)

        print("Training model...")
        metrics_callback = MetricsCallback(validation_data=(
            {"word_level_input": X_val_word}, y_val
        ))

        model.fit(
            {"word_level_input": X_train_word},
            y_train,
            epochs=10,
            batch_size=128,
            validation_data=({"word_level_input": X_val_word}, y_val),
            callbacks=[metrics_callback],
            verbose=1
        )

        print("Evaluating model...")
        loss, accuracy = model.evaluate(
            {"word_level_input": X_test_word},
            y_test
        )

    print(f"Test Accuracy: {accuracy:.2f}")

    print("Saving model...")
    model.save("./model/lstm_model_with_features")
    print("Model saved!")

    print("Saving tokenizer...")
    tokenizer_json = tokenizer.to_json()
    with open("./model/tokenizer.json", "w") as f:
        f.write(tokenizer_json)
    print("Tokenizer saved!")


if __name__ == "__main__":
    main_lstm()