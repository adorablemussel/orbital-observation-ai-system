import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    features_file = "data/processed/model_features.csv"
    labels_file = "data/processed/model_labels.csv"

    # Task 1: Loading Feature Dataset
    X_raw, y_raw = [], []
    feature_names = []
    
    with open(features_file, 'r') as f:
        reader = csv.reader(f)
        feature_names = next(reader)
        for row in reader:
            X_raw.append([float(val) for val in row])
            
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            y_raw.append(int(row[0]))

    print("\n=== Machine Learning: Loading Feature Dataset ===")
    print(f"Input file: {features_file}")
    print(f"Records loaded: {len(X_raw)}")
    print(f"Columns: {feature_names}")

    print("\n=== Machine Learning: Preparing Features and Target ===")
    print(f"Number of samples in X: {len(X_raw)}")
    print(f"Number of labels in y: {len(y_raw)}")
    print(f"Target values detected: {list(set(y_raw))}")

    # Task 3: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    print("\n=== Machine Learning: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Task 4: Training Baseline Model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("\n=== Machine Learning: Model Training ===")
    print("Model: DecisionTreeClassifier\nTraining completed successfully.")

    # Task 5: Generating Predictions
    predictions = model.predict(X_test)
    print("\n=== Machine Learning: Prediction ===")
    print(f"Number of predictions: {len(predictions)}")

    # Task 6: Model Evaluation
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    print("\n=== Machine Learning: Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Task 7: Saving and Inspecting
    import os
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, "results/decision_tree_model.joblib")
    
    print("\n=== Machine Learning: Saving and Inspecting Model ===")
    print("Saved model: results/decision_tree_model.joblib")
    print("Model type: DecisionTreeClassifier")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print("Decision Tree Rules:\n" + export_text(model, feature_names=feature_names))

    # Task 8: Saving Evaluation
    eval_text = (
        f"OOAIS Model Evaluation\n=======\n"
        f"Model: DecisionTreeClassifier\n"
        f"Training samples: {len(X_train)}\n"
        f"Test samples: {len(X_test)}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    with open("results/model_evaluation.txt", "w") as f:
        f.write(eval_text)
    print("\n=== Machine Learning: Saving Evaluation Results ===")
    print("Saved file: results/model_evaluation.txt")

    # Task 9: Training Report
    report_text = (
        f"OOAIS Model Training Summary\n==\n"
        f"Input datasets:\n{features_file}\n{labels_file}\n\n"
        f"Dataset statistics\nNumber of samples: {len(X_raw)}\nNumber of features: {len(feature_names)}\n\n"
        f"Model\nDecisionTreeClassifier\n\n"
        f"Train/Test split\nTraining samples: {len(X_train)}\nTest samples: {len(X_test)}\n\n"
        f"Evaluation summary\nAccuracy: {acc:.4f}\nConfusion Matrix:\n{cm}\n"
    )
    with open("reports/model_training_summary.txt", "w") as f:
        f.write(report_text)
    print("\n=== Machine Learning: Saving Training Report ===")
    print("Saved file: reports/model_training_summary.txt")

if __name__ == "__main__":
    main()