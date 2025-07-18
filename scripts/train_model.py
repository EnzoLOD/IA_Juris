import sys
import os
from src.training.data_preparation import prepare_data
from src.training.fine_tuning import fine_tune_model
from src.training.evaluation import evaluate_model

def main():
    # Prepare the data for training
    print("Preparing data...")
    train_data, val_data = prepare_data()

    # Fine-tune the model
    print("Fine-tuning the model...")
    model = fine_tune_model(train_data, val_data)

    # Evaluate the model
    print("Evaluating the model...")
    evaluation_results = evaluate_model(model, val_data)

    # Print evaluation results
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()