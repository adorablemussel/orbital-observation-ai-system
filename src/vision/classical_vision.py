from pathlib import Path
from PIL import Image
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((64, 64))
    array = np.array(image) / 255.0
    return array.flatten()

def load_image_split(split_dir):
    X = []
    y = []
    class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
                X.append(features)
                y.append(class_name)
    return np.array(X), np.array(y)

def main():
    DATASET_DIR = Path("data/processed/images")
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    print("=== Loading Classical Vision Dataset ===")
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    print("\n=== Model Comparison (Classical ML) ===")
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} s\n")

if __name__ == "__main__":
    main()