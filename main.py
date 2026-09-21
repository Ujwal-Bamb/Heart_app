# final main.py
from data_loader import load_data
from preprocessing import preprocess_and_split
from train_models import train_models
from evaluate_models import evaluate_models

import mlflow

if __name__ == "__main__":
    # 🎯 Set MLflow Experiment First
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("HeartDiseasePrediction")

    # 🚀 Step 1: Load Raw Data
    df = load_data()
    print("\n First 5 rows of raw dataset:")
    print(df.head())

    # 🧹 Step 2: Preprocess, Clean, and Split Data
    X_train, X_test, y_train, y_test, features, scaler = preprocess_and_split()

    # ✅ Step 3: Display Final Information
    print(f"\n Final Split:")
    print(f" Training Samples: {X_train.shape[0]}")
    print(f" Testing Samples: {X_test.shape[0]}")
    print(f" Features Used: {features}")

    # 🏋️ Step 4: Train all models
    print(" Training models...")
    train_models(X_train, y_train)

    # 📈 Step 5: Evaluate and Log to MLflow
    print(" Evaluating models...")
    results = evaluate_models(X_test, y_test)

    print("\n Pipeline Complete!")