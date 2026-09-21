
# preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pickle

def preprocess_and_split(path='data/heart.csv'):
    # 📥 Load the dataset
    df = pd.read_csv(path)
    print(f"Loaded data with shape: {df.shape}")

    # 🧹 Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df.dropna(inplace=True)
        print(f"🧹 Dropped rows with missing values: {missing_count}")
    else:
        print("No missing values found")

    # 🧹 Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"🧹 Dropped duplicate rows: {duplicate_count}")
    else:
        print("No duplicates found")

    # 🔠 Encode categorical columns (if any)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f" Encoded categorical columns: {categorical_cols}")
    else:
        print("No categorical columns to encode")

    # 🎯 Separate features and target
    if 'HeartDisease' not in df.columns:
        raise ValueError("'HeartDisease' column not found in dataset")

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # 📊 Identify numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = X.columns.difference(numeric_cols)

    # 📏 Apply scaling only on numeric columns
    scaler = StandardScaler()
    X_scaled_numeric = scaler.fit_transform(X[numeric_cols])

    # Convert scaled numeric features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled_numeric, columns=numeric_cols, index=X.index)

    # 🧱 Combine scaled numeric and unscaled non-numeric features
    X_processed = pd.concat([X_scaled_df, X[non_numeric_cols]], axis=1)

    # 💾 Save the scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 🌲 Feature Selection using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_processed, y)

    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_selected = selector.transform(X_processed)

    original_features = X_processed.columns.tolist()
    selected_features = np.array(original_features)[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} important features out of {len(original_features)}")

    # 💾 Save selected features
    with open('models/selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    # ✂️ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, selected_features, scaler