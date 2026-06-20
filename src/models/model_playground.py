from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def validate_input_files():
    f_path = Path("data/processed/model_features.csv")
    l_path = Path("data/processed/model_labels.csv")
    if not f_path.exists() or not l_path.exists():
        print("\n=== Model Playground: Input Validation ===")
        print("Error: missing required input file(s):")
        print("data/processed/model_features.csv\ndata/processed/model_labels.csv")
        exit(1)

def load_data():
    features_df = pd.read_csv("data/processed/model_features.csv")
    labels_df = pd.read_csv("data/processed/model_labels.csv")
    print("\n=== Model Playground: Loading Data ===")
    print("Feature file: data/processed/model_features.csv")
    print("Label file: data/processed/model_labels.csv")
    return features_df, labels_df

def inspect_data(features_df, labels_df):
    if features_df.empty or labels_df.empty or len(features_df) != len(labels_df) or "anomaly_flag" not in labels_df.columns:
        print("\n=== Model Playground: Data Inspection ===")
        print("Error: Data validation failed.")
        exit(1)
    
    print("\n=== Model Playground: Data Inspection ===")
    print(f"Number of samples: {len(features_df)}")
    print(f"Number of features: {features_df.shape[1]}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Target values detected: {list(labels_df['anomaly_flag'].unique())}")

def prepare_features_and_labels(features_df, labels_df):
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values
    print("\n=== Model Playground: Preparing Features and Labels ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\n=== Model Playground: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def define_models():
    return {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

def train_models(models, X_train, y_train):
    trained_models = {}
    print("\n=== Model Playground: Training Models ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name}: trained")
        trained_models[name] = model
    return trained_models

def generate_predictions(trained_models, X_test):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results.append({"name": name, "model": model, "y_pred": y_pred})
    return results

def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("\n=== Model Playground: Example Predictions ===")
    for i in range(min(num_examples, len(y_test))):
        line = f"True: {y_test[i]}"
        for res in prediction_results:
            line += f" | {res['name']}: {res['y_pred'][i]}"
        print(line)

def compute_accuracy(prediction_results, y_test):
    print("\n=== Model Playground: Accuracy Comparison ===")
    for res in prediction_results:
        acc = accuracy_score(y_test, res["y_pred"])
        res["accuracy"] = acc
        print(f"{res['name']}: {acc:.4f}")
    return prediction_results

def compute_detailed_metrics(prediction_results, y_test):
    print("\n=== Model Playground: Detailed Evaluation ===")
    for res in prediction_results:
        y_pred = res["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        res["confusion_matrix"] = cm
        res["classification_report"] = report
        
        print(f"\nModel: {res['name']}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Class labels:\n0 -> normal observation\n1 -> anomaly\n")
        print("Classification Report:\n-----------------------------------------")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        print("-----------------------------------------")
        for cls in ['0', '1']:
            if cls in report:
                print(f"{cls:<20} {report[cls]['precision']:<10.2f} {report[cls]['recall']:<10.2f} {report[cls]['f1-score']:<10.2f} {int(report[cls]['support']):<10}")
        print("-----------------------------------------")
    return prediction_results

def rank_models(evaluation_results):
    sorted_results = sorted(evaluation_results, key=lambda x: x["accuracy"], reverse=True)
    print("\n=== Model Playground: Ranking ===")
    for idx, res in enumerate(sorted_results, start=1):
        print(f"{idx}. {res['name']} - {res['accuracy']:.4f}")
    return sorted_results

def run_controlled_experiments(X_train, y_train, X_test, y_test):
    print("\n=== Model Playground: Controlled Experiments ===")
    results = []
    depths = [2, 3, 5]
    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        name = f"Decision Tree (max_depth={d})"
        print(f"{name}: {acc:.4f}")
        results.append({"name": name, "accuracy": acc})
    
    estimators = [5, 10, 50]
    for n in estimators:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        name = f"Random Forest (n_estimators={n})"
        print(f"{name}: {acc:.4f}")
        results.append({"name": name, "accuracy": acc})
        
    return results

def save_experiment_summary(features_path, labels_path, X, X_train, X_test, ranked_models, experiment_results):
    summary_path = "reports/model_playground_summary.txt"
    with open(summary_path, "w") as f:
        f.write("OOAIS Model Playground Summary\n==============================\n\n")
        f.write(f"Input datasets\n{features_path}\n{labels_path}\n\n")
        f.write(f"Dataset statistics\nNumber of samples: {X.shape[0]}\nNumber of features: {X.shape[1]}\n")
        f.write(f"Training samples: {len(X_train)}\nTesting samples: {len(X_test)}\n\n")
        f.write("Compared models\n")
        for r in ranked_models:
            f.write(f"{r['name']}: {r['accuracy']:.4f}\n")
        f.write(f"\nBest model\n{ranked_models[0]['name']} achieved the highest accuracy: {ranked_models[0]['accuracy']:.4f}\n\n")
        f.write("Additional experiments\n")
        for r in experiment_results:
            f.write(f"{r['name']}: {r['accuracy']:.4f}\n")
        f.write("\nConclusion\nThe best candidate for further experiments is clearly visible above.\n")
    print("\n=== Model Playground: Saving Summary ===")
    print(f"Saved file: {summary_path}")

def create_metric_plots(ranked_models):
    names = [r["name"] for r in ranked_models]
    accs = [r["accuracy"] for r in ranked_models]
    
    precs, recs, f1s = [], [], []
    for r in ranked_models:
        cls_1 = r["classification_report"].get("1", {"precision": 0, "recall": 0, "f1-score": 0})
        precs.append(cls_1["precision"])
        recs.append(cls_1["recall"])
        f1s.append(cls_1["f1-score"])
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].bar(names, accs)
    axes[0,0].set_title("Accuracy")
    axes[0,1].bar(names, precs)
    axes[0,1].set_title("Precision (Anomaly)")
    axes[1,0].bar(names, recs)
    axes[1,0].set_title("Recall (Anomaly)")
    axes[1,1].bar(names, f1s)
    axes[1,1].set_title("F1-score (Anomaly)")
    
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        
    plt.tight_layout()
    plt.savefig("reports/model_comparison_panel.png")
    plt.close()
    print("\n=== Model Playground: Saving Visualizations ===")
    print("Saved file: reports/model_comparison_panel.png")

def main():
    validate_input_files()
    f_df, l_df = load_data()
    inspect_data(f_df, l_df)
    X, y = prepare_features_and_labels(f_df, l_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = define_models()
    trained_models = train_models(models, X_train, y_train)
    predictions = generate_predictions(trained_models, X_test)
    
    print_example_predictions(predictions, y_test)
    predictions = compute_accuracy(predictions, y_test)
    predictions = compute_detailed_metrics(predictions, y_test)
    ranked_models = rank_models(predictions)
    
    exp_results = run_controlled_experiments(X_train, y_train, X_test, y_test)
    
    save_experiment_summary("data/processed/model_features.csv", "data/processed/model_labels.csv", 
                            X, X_train, X_test, ranked_models, exp_results)
    create_metric_plots(ranked_models)

if __name__ == "__main__":
    main()